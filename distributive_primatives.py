import os
import process_group_manager as pgm
import torch, torch.distributed as dist
import process_group_manager as pgm


def communicate(operation, device, dtype, tensor=None, shapes=None):
    src, dst = None, None
    
    if operation=="recv_forward":
        if pgm.process_group_manager.pp_is_first_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank
    elif operation=="send_forward":
        if pgm.process_group_manager.pp_is_last_stage: return
        dst = pgm.process_group_manager.pp_next_rank
    elif operation=="recv_backward":
        if pgm.process_group_manager.pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank
    elif operation=="send_backward":
        if pgm.process_group_manager.pp_is_first_stage: return
        dst = pgm.process_group_manager.pp_prev_rank
    
    is_send = operation.startswith('send')
    peer_rank = dst if is_send else src
    
    if peer_rank is None:
        return None
    
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    reqs = dist.batch_isend_irecv([op])
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    return tensor if not is_send else None

def bidirectional_communicate(operation, send_tensor, recv_shape, device, dtype):
    is_fwd = (operation == "send_fwd_recv_bwd")  # Fixed typo here
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or (not is_fwd and pgm.process_group_manager.pp_is_first_stage): 
        return None
    
    peer_rank = pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank
    recv_tensor = torch.empty(recv_shape, requires_grad=True, device=device, dtype=dtype)
    send_op = dist.P2POp(dist.isend, send_tensor, peer_rank)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, peer_rank)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    return recv_tensor