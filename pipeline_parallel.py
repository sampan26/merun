import torch, torch.nn as nn, torch.nn.functional as F
from distributive_primatives import communicate, bidirectional_communicate
import process_group_manager as pgm
import warnings
import torch.distributed as dist


warnings.filterwarnings("ignore", message=".*`resume_download` is deprecated.*")

def reduce_loss_across_dp_ranks(loss, device):
    reduced_loss = torch.tensor([loss if loss is not None else 0.0], dtype=torch.float32, device=device)
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.world_group)
    reduced_loss /= pgm.process_group_manager.dp_world_size
    return reduced_loss.item()

class PipelineParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        layer_distribution = self.distribute_layers(config.num_hidden_layers)
        self.embed_tokens = model.model.embed_tokens if pgm.process_group_manager.pp_is_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): model.model.layers[i] for i in layer_distribution})
        self.norm = model.model.norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        self.lm_head = model.lm_head if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        del model

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // pgm.process_group_manager.pp_world_size + (1 if i < num_layers % pgm.process_group_manager.pp_world_size else 0) for i in range(pgm.process_group_manager.pp_world_size)]
        start_layer = sum(layers_per_gpu[:pgm.process_group_manager.pp_rank])
        return list(range(start_layer, start_layer+layers_per_gpu[pgm.process_group_manager.pp_rank]))
    
    def forward(self, batch, device):
        x = batch["hidden_states"].to(device) if batch["hidden_states"] is not None else batch["input_ids"].to(device)
        x = self.embed_tokens(x)
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=batch['position_idx'].to(device))[0]
        x = self.norm(x)
        return self.lm_head(x)
    
    def backwards(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

def pipeline_parallel_1f1b(model, dataloader, tensor_shape, device):
    num_warmup_microbatches = min(dataloader.num_local_micro_batches, pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank  - 1)
    num_remaining_microbatches = dataloader.num_local_micro_batches - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors = 0.0, [], []

    def _forward_step(input_tensor):
        batch = next(iter(dataloader))
        batch['hidden_states'] = input_tensor
        output_tensor = model(batch, device)
        if pgm.process_group_manager.pp_is_last_stage and pgm.process_group_manager.global_rank == pgm.process_group_manager.tp_first_rank:
            loss = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += loss.item()
            return loss
        return output_tensor

    # Warmup phase
    for _ in range(num_warmup_microbatches):
        input_tensor = communicate('recv_forward', shapes=tensor_shape, device=device, dtype=torch.float32)
        output_tensor = _forward_step(input_tensor)
        communicate("send_forward", output_tensor)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
    
    # 1F1B phase
    if num_remaining_microbatches > 0:
        input_tensor = communicate(operation='recv_forward', shapes=tensor_shape, device=device, dtype=torch.float32)

    for i in range(num_remaining_microbatches):
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_communicate(
            operation="send_fwd_recv_bwd", 
            send_tensor=output_tensor,  # Use the tensor, not the list
            recv_shape=tensor_shape, 
            device=device, 
            dtype=torch.float32
        )
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        
        # Process the oldest saved tensors
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backwards(input_tensor, output_tensor, output_tensor_grad)
        
        if i == num_remaining_microbatches - 1:
            input_tensor = None
            communicate('send_backward', tensor=input_tensor_grad, device=device)
        else:
            input_tensor = bidirectional_communicate(
                operation="send_bwd_recv_fwd", 
                send_tensor=input_tensor_grad,
                recv_shape=tensor_shape, 
                device=device, 
                dtype=torch.float32
            )
    
    # Cooldown phase
    for _ in range(num_warmup_microbatches):
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = communicate('recv_backward', shapes=tensor_shape, device=device, dtype=torch.float32)
        input_tensor_grad = model.backwards(input_tensor, output_tensor, output_tensor_grad)
        communicate("send_backward", tensor=input_tensor_grad, device=device, dtype=torch.float32)

    
    logging_loss = reduce_loss_across_dp_ranks(logging_loss, device)
    return logging_loss