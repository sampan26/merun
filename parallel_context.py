import torch.distributed as dist

class ParallelContext:
    def __init__(self, pp_rank, pp_world_size):
        self.pp_rank, self.pp_world_size = pp_rank, pp_world_size
        self.pp_group = dist.new_group(ranks=list(range(self.pp_world_size)))
        self.pp_next_rank = None if self.pp_rank == self.pp_world_size - 1 else (self.pp_rank + 1) % self.pp_world_size
        self.pp_prev_rank = None if self.pp_rank == 0 else (self.pp_rank - 1) % self.pp_world_size
        self.is_pipeline_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_last_rank = self.pp_world_size - 1
        self.is_pipeline_first_stage = self.pp_rank == 0

def setup_parallel_context(local_rank, world_size):
    global parallel_context
    parallel_context = ParallelContext(local_rank, world_size)
