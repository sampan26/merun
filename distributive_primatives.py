import os
import parallel_context as pc
import torch, torch.distributed as dist
import parallel_context as pc


def communicate(operation='send_forward', tensor=None, shapes=None, dtype=None):
    src, dst = None, None
    
    if operation=="recv_forward":
        if pc.parallel_context.pp_is_first_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device="cuda", dtype=dtype)
        src = pc.parallel_context.pp_prev_rank
    elif operation=="send_forward":
        if pc.parallel_context.pp_is_last_stage: return
        dst = pc.parallel_context.pp_next_rank
    elif operation=="recv_backward":
        if pc.parallel_context.pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device="cuda", dtype=dtype)
        src = pc.parallel_context.pp_next_rank
    elif operation=="send_backward":
        if pc.parallel_context.pp_is_first_stage: return
        dst = pc.parallel_context.pp_prev_rank
    
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
    if (is_fwd and pc.parallel_context.pp_is_last_stage) or (not is_fwd and pc.parallel_context.pp_is_first_stage): 
        return None
    
    peer_rank = pc.parallel_context.pp_next_rank if is_fwd else pc.parallel_context.pp_prev_rank
    recv_tensor = torch.empty(recv_shape, requires_grad=True, device=device, dtype=dtype)
    send_op = dist.P2POp(dist.isend, send_tensor, peer_rank)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, peer_rank)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    return recv_tensor