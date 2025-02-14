import os
import torch
import torch.distributed as dist

class ParallelContext:
    def __init__(self, tp_size, pp_size, dp_size):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))

        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        assert self.world_size == self.tp_size * self.pp_size * self.dp_size,  f"World size ({self.world_size}) != TP ({self.tp_size}) * PP ({self.pp_size}) * DP ({self.dp_size})"
        self.grid = torch.arange(self.world_size).view(self.pp_size, self.dp_size, self.tp_size).permute(2, 0, 1)
        self.tp_rank, self.pp_rank, self.dp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()

        self.tp_group_ids = self.grid[:, self.pp_rank, self.dp_rank].tolist()
        self.pp_group_ids = self.grid[self.tp_rank, :, self.dp_rank].tolist()
        self.dp_group_ids = self.grid[self.tp_rank, self.pp_rank, :].tolist()
        self.tp_pp_group_ids = self.grid[..., self.dp_rank].tolist()

        self.tp_group = dist.new_group(self.tp_group_ids)
        self.pp_group = dist.new_group(self.pp_group_ids)
        self.dp_group = dist.new_group(self.dp_group_ids)
        self.tp_pp_group = dist.new_subgroups_by_enumeration(self.tp_pp_group_ids)[0]

        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]
        self.tp_is_first_stage = self.tp_rank == 0
        self.tp_is_last_stage = self.tp_rank == self.tp_size - 1
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
        self.dp_is_first_stage = self.dp_rank == 0
        self.dp_is_last_stage = self.dp_rank == self.dp_size - 1
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        
        self.tp_pp_world_size = dist.get_world_size(group=self.tp_pp_group)
    
    def __str__(self):
        return f"DP({self.dp_size})-PP({self.pp_size})-TP({self.tp_size})-Rank({self.global_rank})"
    
    def display_parallelism_grid(self):
        def _create_box(content):
            return f"  {content:^3}  "
        def _create_row(row):
            return "|" + "|".join(_create_box(f"g{num:02d}") for num in row) + "|"
        def _create_border(width):
            return "+" + "-" * (width - 2) + "+"
        def _create_pp_line(width, pp_size):
            box_width = (width - pp_size + 1) // pp_size
            return "  ".join("PP".center(box_width) for _ in range(pp_size))
        output = []
        sample_row = _create_row(self.grid[0, :, 0])
        row_width = len(sample_row)
        border = _create_border(row_width)
        output.append(f"=== Global Parallelism Configuration ===")
        output.append(f"DP Size: {self.dp_size}, PP Size: {self.pp_size}, TP Size: {self.grid.shape[0]}")
        output.append("")  # Top spacing
        for dp in range(self.dp_size):
            output.append(f"DP {dp}:")
            output.append(f"{'':>8}{border}")
            
            for tp in range(self.grid.shape[0]):
                if tp == 0:
                    output.append(f"{'TP':>7} {_create_row(self.grid[tp, :, dp])}")
                else:
                    output.append(f"{'':8}{border}")
                    output.append(f"{'TP':>7} {_create_row(self.grid[tp, :, dp])}")
            
            output.append(f"{'':8}{border}")
            if self.pp_size > 1:
                output.append(f"{'':>7}{_create_pp_line(row_width, self.pp_size)}")
            output.append("")  # Spacing between DP blocks
        output.append("")  # Bottom spacing
        output.append(f"=== Local Parallelism Configuration ===")
        output.append(self.__str__())            
        output.append(f"DP Group IDs: {['g{:02d}'.format(id) for id in self.dp_group_ids]}")
        output.append(f"PP Group IDs: {['g{:02d}'.format(id) for id in self.pp_group_ids]}")
        output.append(f"TP Group IDs: {['g{:02d}'.format(id) for id in self.tp_group_ids]}")
        output.append(f"TP-PP Group IDs: {[['g{:02d}'.format(id) for id in subgroup] for subgroup in self.tp_pp_group_ids]}")
        print("\n".join(output))


def setup_parallel_context(tp_size, pp_size, dp_size):
    global parallel_context
    parallel_context = ParallelContext(tp_size, pp_size, dp_size)
