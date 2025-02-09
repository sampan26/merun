from transformers import AutoConfig, AutoModelForCausalLM
import torch, torch.nn as nn, torch.functional as F
from distributive_primatives import communicate, bidirectional_communicate
import parallel_context as pc

class PipelineParallel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
        layer_distribution = self.distribute_layers(self.config.num_layers)
        self.embed_tokens = base_model.model.embed_tokens if pc.parallel_context.is_pipeline_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): base_model.model.layers[i] for i in layer_distribution})
        self.norm = base_model.model.norm if pc.parallel_context.is_pipeline_last_stage else nn.Identity()
        self.lm_head = base_model.model.lm_head if pc.parallel_context.is_pipeline_last_stage else nn.Identity()
        del base_model

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // pc.parallel_context.pp_world_size + (1 if i < num_layers % pc.parallel_context.pp_world_size else 0) for i in range(pc.parallel_context.pp_world_size)]
        start_layer = sum(layers_per_gpu[:pc.parallel_context.pp_rank])
        return list(range(start_layer, start_layer+layers_per_gpu[pc.parallel_context.pp_rank]))
    
    def forward(self, batch, device):
        x = batch["hidden_states"].to(device) if batch["hidden_states"] is not None else batch["input_ids"].to(device)
        x = self.embed_tokens(x)
        for layer in self.decoder_layers:
            x = layer(x, position_ids=batch['position_idx'].to(device))[0]
        x = self.norm(x)
        return self.lm_head(x)
    
    def backwards(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backwards(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None


def pipeline_parallel_1f1b(model, dataloader, tensor_shape, device):
    num_warmup_microbatches = min(dataloader.num_local_micro_batches, pc.parallel_context.pp_world_size - pc.parallel_context.pp_rank  - 1)
    num_remaining_microbataches = dataloader.num_local_micro_batches - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors = 0.0, [], []

    def _forward_step(input_tensor):
        batch = next(iter(dataloader))
        batch['hidden_states'] = input_tensor
        output_tensors = model(batch, device)
        if pc.parallel_context.is_pipeline_last_stage:
            output_tensors = F.cross_entropy(output_tensor.transpose(1,2), batch['target_ids'].to(device), reduction="mean")
            nonlocal logging_loss
            logging_loss += output_tensor.item()
        return output_tensors

    for _ in range(num_warmup_microbatches):
        input_tensor = communicate('recv_forward', shapes=tensor_shape, dtype=torch.float32)
        output_tensors = _forward_step(input_tensor)
        communicate("send_forward", output_tensors)
        input_tensors.append(input_tensors)
        output_tensors.append(output_tensors)
    
    if num_remaining_microbataches > 0:
        input_tensor = communicate(operation='recv_forward', shapes=tensor_shape, dtype=torch.float32)

    for i in range(num_remaining_microbataches):
        output_tensors = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_communicate(operation="send_fwd_recv_bwd", send_tensor=output_tensors, recv_shape=tensor_shape, device=device, dtype=torch.float32)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensors)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backwards(input_tensor, output_tensor, output_tensor_grad)
        if i == num_remaining_microbataches - 1:
            input_tensor = None
            communicate('send_backward', tensor=input_tensor_grad)
        else:
            input_tensor = bidirectional_communicate(operation="send_bwd_recv_fwd", send_tensor=input_tensor_grad, recv_shape=tensor_shape, device=device, dtype=torch.float32)
    
    for _ in range(num_warmup_microbatches):
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = communicate('recv_backward', shapes=tensor_shape, dtype=torch.float32)
        input_tensor_grad = model.backwards(input_tensor, output_tensor, output_tensor_grad)
        communicate("send_backward", tensor=input_tensor_grad)
    