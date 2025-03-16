import torch
import random
import numpy as np
import builtins
import fcntl
import process_group_manager as pgm

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def display_parallelism_grid():
    def _create_gpu_box(gpu_num, tp, cp, pp, dp):
        return [
            f"      GPU {gpu_num:<2}   ",
            f"  +----------+",
            f"  | tp{tp} cp{cp}  |",
            f"  | pp{pp} dp{dp}  |",
            f"  +----------+"
        ]

    def _create_row(start_gpu, tp_size, cp, pp, dp):
        boxes = [_create_gpu_box(start_gpu + i, i, cp, pp, dp) for i in range(tp_size)]
        return [" ".join(row) for row in zip(*boxes)]

    def _add_pp_label(output):
        output.append("   |  ")
        output.append(" PP|  ")
        output.append("   |  ")

    def _add_cp_label(output):
        output.append("   |  CP")

    def _add_vertical_separator(output):
        output.append("   |  ")
        output.append("   |  |")

    def _add_vertical_arrow(output):
        output.append("   |  v")

    def _add_horizontal_separator(output):
        output.append("-" * 86)

    def _create_tp_arrows_and_labels(tp_group_width):
        tp_arrow = "-" * (tp_group_width - 4) + ">"
        tp_label = f"{'TP':^{tp_group_width}}"
        tp_arrows = f"         {tp_arrow:<{tp_group_width}}           {tp_arrow}"
        tp_labels = f"         {tp_label:<{tp_group_width}}           {tp_label}"
        return tp_arrows, tp_labels

    def _create_dp_arrow_and_label(total_tp_width):
        dp_arrow = "-" * (total_tp_width - 6) + ">"
        dp_label = f"{'DP':^{total_tp_width}}"
        return f"      {dp_arrow}", f"      {dp_label}"

    output = []
    tp_size = pgm.process_group_manager.tp_size
    cp_size = pgm.process_group_manager.cp_size
    pp_size = pgm.process_group_manager.pp_size
    dp_size = pgm.process_group_manager.dp_size

    output.append("=== Global Parallelism Configuration ===")
    output.append(f"TP Size: {tp_size}, CP_size: {cp_size}, PP Size: {pp_size}, DP Size: {dp_size}")
    output.append("")

    for dp in range(0, dp_size, 2):
        output.append("   |  ")

        for pp in range(pp_size):
            if pp == pp_size // 2:
                _add_pp_label(output)
            
            _add_vertical_separator(output)
            
            for cp in range(cp_size):
                left_start_gpu = dp * (tp_size * cp_size * pp_size) + pp * (tp_size * cp_size) + cp * tp_size
                left_row = _create_row(left_start_gpu, tp_size, cp, pp, dp)

                if dp + 1 < dp_size:
                    right_start_gpu = (dp+1) * (tp_size * cp_size * pp_size) + pp * (tp_size * cp_size) + cp * tp_size
                    right_row = _create_row(right_start_gpu, tp_size, cp, pp, dp+1)
                    for l, r in zip(left_row, right_row):
                        output.append(f"   |  | {l:<33}  {r}")
                else:
                    for l in left_row:
                        output.append(f"   |  | {l}")

                if cp < cp_size - 1:
                    _add_cp_label(output)
                output.append("   |  |")

            _add_vertical_arrow(output)

            if pp < pp_size - 1:
                output.append("   |  ")

        output.append("   |  ")
        output.append("   v  ")

        if dp + 2 < dp_size:
            _add_horizontal_separator(output)

    tp_group_width = tp_size * 13 - 1
    total_tp_width = tp_group_width * 2 + 18

    tp_arrows, tp_labels = _create_tp_arrows_and_labels(tp_group_width)
    dp_arrow, dp_label = _create_dp_arrow_and_label(total_tp_width)

    output.extend(["", tp_arrows, tp_labels, "", dp_arrow, dp_label])

    print("\n".join(output))