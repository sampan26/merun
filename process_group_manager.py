import os
import torch
import torch.distributed as dist

class ProcessGroupManager:
    def __init__(self, tp_size, pp_size, dp_size):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))

        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        assert self.world_size == self.tp_size * self.pp_size * self.dp_size,  f"World size ({self.world_size}) != TP ({self.tp_size}) * PP ({self.pp_size}) * DP ({self.dp_size})"
        self.grid = torch.arange(self.world_size).view(self.tp_size, self.pp_size, self.dp_size)
        self.tp_rank, self.pp_rank, self.dp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()

        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[i, j, :].tolist() for i in range(tp_size) for j in range(pp_size)])[0]
        self.tp_group = dist.new_subgroups_by_enumeration([self.grid[:, i, j].tolist() for i in range(pp_size) for j in range(dp_size)])[0]
        self.pp_group = dist.new_subgroups_by_enumeration([self.grid[i, :, j].tolist() for i in range(tp_size) for j in range(dp_size)])[0]
        self.world_group = dist.group.WORLD

        self.tp_group_ids = self.grid[:, self.pp_rank, self.dp_rank].tolist()
        self.pp_group_ids = self.grid[self.tp_rank, :, self.dp_rank].tolist()
        self.dp_group_ids = self.grid[self.tp_rank, self.pp_rank, :].tolist()

      
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]
        self.tp_world_size = dist.get_world_size(group=self.tp_group)

        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_size - 1
        self.pp_next_rank = None if self.pp_rank == self.pp_size - 1 else int(self.grid[self.tp_rank, self.pp_rank + 1, self.dp_rank].item())
        self.pp_prev_rank = None if self.pp_rank == 0 else int(self.grid[self.tp_rank, self.pp_rank - 1, self.dp_rank].item())
        self.pp_world_size = dist.get_world_size(group=self.pp_group)
        
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        
    
    def __str__(self):
        return f"DP({self.dp_size})-PP({self.pp_size})-TP({self.tp_size})-Rank({self.global_rank})"
    
def setup_process_group_manager(tp_size, pp_size, dp_size):
    global process_group_manager
    process_group_manager = ProcessGroupManager(tp_size, pp_size, dp_size)
