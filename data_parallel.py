import torch.distributed as dist
import torch.nn as nn
import process_group_manager as pgm

class DataParallel(nn.Module):
    def __init__(self, model, config):
        #TODO: Add Zero1
        #TODO: Interleave all_reduce
        super().__init__()
        self.model = model
        self.dp_world_size = pgm.process_group_manager.dp_world_size
        self.dp_rank = pgm.process_group_manager.dp_rank

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.model.backward(input_tensor, output_tensor, output_tensor_grad)

    def all_reduce_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=pgm.process_group_manager.dp_group)
                