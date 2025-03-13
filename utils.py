import torch, random, numpy as np
import process_group_manager as pgm

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def display_parallelism_grid():
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
    sample_row = _create_row(pgm.process_group_manager.grid[0, :, 0])
    row_width = len(sample_row)
    border = _create_border(row_width)

    output.append(f"=== Global Parallelism Configuration ===")
    output.append(f"DP Size: {pgm.process_group_manager.dp_size}, PP Size: {pgm.process_group_manager.pp_size}, TP Size: {pgm.process_group_manager.grid.shape[0]}")
    output.append("")  # Top spacing

    for dp in range(pgm.process_group_manager.dp_size):
        output.append(f"DP {dp}:")
        output.append(f"{'':>8}{border}")
        
        for tp in range(pgm.process_group_manager.grid.shape[0]):
            if tp == 0:
                output.append(f"{'TP':>7} {_create_row(pgm.process_group_manager.grid[tp, :, dp])}")
            else:
                output.append(f"{'':8}{border}")
                output.append(f"{'TP':>7} {_create_row(pgm.process_group_manager.grid[tp, :, dp])}")
        
        output.append(f"{'':8}{border}")
        if pgm.process_group_manager.pp_size > 1:
            output.append(f"{'':>7}{_create_pp_line(row_width, pgm.process_group_manager.pp_size)}")

        output.append("")  # Spacing between DP blocks

    output.append("")  # Bottom spacing

    output.append(f"=== Local Parallelism Configuration ===")
    output.append(pgm.process_group_manager.__str__())            
    output.append(f"DP Group IDs: {['g{:02d}'.format(id) for id in pgm.process_group_manager.dp_group_ids]}")
    output.append(f"PP Group IDs: {['g{:02d}'.format(id) for id in pgm.process_group_manager.pp_group_ids]}")
    output.append(f"TP Group IDs: {['g{:02d}'.format(id) for id in pgm.process_group_manager.tp_group_ids]}")

    print("\n".join(output))