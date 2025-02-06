import os
import torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_datasets
from transformers import AutoTokenizer
from parallel_context import setup_parallel_context
from pipeline_parallel import PipelineParallel


class MicroBatchDataLoader(DataLoader):
    def __init__(self, global_batch_size, micro_batch_size, seq_len, data_parallel_size, dataset_name, tokenizer, split="train", num_samples=None):
        self.global_batch_size, self.micro_batch_size, self.seq_length, self.data_parallel_size = global_batch_size, micro_batch_size, seq_len, data_parallel_size
        self.local_batch_size = self.global_batch_size // self.data_parallel_size
        self.num_local_micro_batches = self.local_batch_size // self.micro_batch_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        self.dataset = load_datasets(dataset_name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if num_samples:
            self.dataset.select(range(min(num_samples, len(self.dataset))))
        dist.barrier()
        self.dataset = self.tokenize_dataset(self.dataset)
        super().__init__(self.dataset, batch_size=micro_batch_size, pin_memory=True, num_workers=3, collate_fn=self.collate_fn, sampler=DistributedSampler(self.dataset, shuffle=False, num_replicas=data_parallel_size), shuffle=False)

    def collate_fn(self, batch_data):
        batch_input_ids = torch.stack([data['input_ids'] for data in batch_data])
        batch_size, seq_len = batch_input_ids.shape
        return {
            'input_ids': batch_input_ids[: , -1].T.contiguous,
            'target_ids': batch_input_ids[: , 1:].T.contiguous,
            'position_idx': torch.arange(seq_len - 1, dtype=torch.long).unsqueeze(-1).expand(-1, batch_size).contiguous(),
            'attention_mask': torch.tril(torch.ones((seq_len - 1, seq_len - 1), dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1).contigous(),
            'hidden_state': None
        }

    def tokenize_dataset(self, dataset):
        return dataset.map(
            lambda example: self.tokenizer(example["text"], 
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.seq_length+1,
                                           return_special_tokens_mask=False),
            batched=True,
            remove_columns=dataset.column_names
        ).with_format("torch", columns=["input_ids"])

if __name__ == "__main__":
    local_rank, world_size = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_RANK'])

    GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LEN, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS = 6, 2, 10, 1e-4, 20, 1800
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    setup_parallel_context(local_rank, world_size)

    
    model = PipelineParallel("HuggingFaceTB/SmolLM-360M-Instruct").to(device)
