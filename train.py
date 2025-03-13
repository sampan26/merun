import os
import torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM,AutoTokenizer

import process_group_manager as pgm 
from process_group_manager import setup_process_group_manager
from utils import set_all_seed, display_parallelism_grid
from pipeline_parallel import PipelineParallel, pipeline_parallel_1f1b
from data_parallel import DataParallel
from dataset import MicroBatchDataLoader
import argparse

def train_step(model, data_loader, device):
    total_loss = 0.0

    for _ in range(data_loader.num_local_micro_batches):
        batch = next(iter(data_loader))
        input_ids = batch['input_ids'].to(device)
        position_ids = batch["position_index"].to(device)
        target_ids = batch["target_ids"].to(device)

        outputs = model(input_ids=input_ids, position_ids=position_ids)
        logits = outputs.logits

        loss = F.cross_entropy(logits.transpose(1,2), target_ids, reduction="mean")
        loss.backward()

        total_loss += loss.item()
    
    avg_loss = total_loss / data_loader.num_local_micro_batches
    return avg_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank, world_size = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    host, port = os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])

    dist.init_process_group(rank=local_rank, world_size=world_size, backend="nccl", init_method=f"tcp://{host}:{port}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    setup_parallel_context(tp_size=args.tp_size, pp_size=args.pp_size, dp_size=args.dp_size)
    
    if pgm.process_group_manager.pp_is_first_stage and pgm.process_group_manager.global_rank == pgm.process_group_manager.dp_first_rank:
        display_parallelism_grid()

    
    GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LEN, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS, SEED = 6, 2, 10, 1e-4, 20, 1800, 42
    
    set_all_seed(seed=SEED)

    data_path = "/ib-scratch/chenguang03/scratch/pan.samuel/misc/merun/tiny_stories"
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    config = AutoConfig.from_pretrained(model_name)

    if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
         wandb.init(
             project="merun",
             name=f"test_convergence_{pgm.process_group_manager}",
             config={
                 "data_parallel_size": pgm.process_group_manager.dp_size,
                 "tensor_parallel_size": pgm.process_group_manager.tp_size,
                 "pipeline_parallel_size": pgm.process_group_manager.pp_size,
                 "model": model_name,
                 "dataset": dataset_name,
                 "max_tokens": MAX_TOKENS,
                 "learning_rate": LEARNING_RATE,
                 "seed": SEED,
                 "micro_batch_size": MICRO_BATCH_SIZE,
                 "global_batch_size": GLOBAL_BATCH_SIZE,
             },
         )
     

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

    if pgm.process_group_manager.pp_world_size > 1:
        model = PipelineParallel(model, config).to(device)
    
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallel(model, config).to(device)

    model.train()
     
    data_loader = MicroBatchDataLoader(GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LEN, data_path, model_name, num_samples=NUM_SAMPLES)
    tensor_shapes = (SEQ_LEN, data_loader.micro_batch_size, config.hidden_size)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    trained_tokens, step = 0, 0
    tokens_per_step = GLOBAL_BATCH_SIZE * SEQ_LEN

    dist.barrier()

    while trained_tokens < MAX_TOKENS:
        data_loader.set_epoch(step)

        optimizer.zero_grad()
        if pgm.process_group_manager.pp_world_size > 1:
            loss = pipeline_parallel_1f1b(model, data_loader, tensor_shapes, device)
        else:
            loss = train_step(model, data_loader, device)
        
        if pgm.process_group_manager.dp_world_size > 1:
             # Average gradient across DP ranks
            model.all_reduce_gradients()

        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        if pgm.process_group_manager.global_rank == 0:
            print(f"[rank {pgm.process_group_manager.global_rank}] Step: {step}, Loss: {loss:.4f}, Tokens: {trained_tokens}/{MAX_TOKENS}")
        
        if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
            wandb.log({"loss": loss, "trained_tokens": trained_tokens})
            
    if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()