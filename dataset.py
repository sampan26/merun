import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset

import process_group_manager as pgm

class MicroBatchDataLoader(DataLoader):
    def __init__(self, global_batch_size, micro_batch_size, seq_len, dataset_name, tokenizer, split="train", num_samples=None):
        self.global_batch_size, self.micro_batch_size, self.seq_length,  = global_batch_size, micro_batch_size, seq_len
        self.local_batch_size = self.global_batch_size // pgm.process_group_manager.dp_world_size
        self.num_local_micro_batches = self.local_batch_size // self.micro_batch_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        self.dataset = load_from_disk(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if num_samples:
            self.dataset.select(range(min(num_samples, len(self.dataset))))
        dist.barrier()
        self.dataset = self.tokenize_dataset(self.dataset)
        
        self.sampler = DistributedSampler(self.dataset, num_replicas=pgm.process_group_manager.dp_world_size, rank=pgm.process_group_manager.dp_rank, shuffle=False)
        super().__init__(self.dataset, batch_size=micro_batch_size, collate_fn=self.collate_batch, pin_memory=True, num_workers=3, sampler=self.sampler, shuffle=False)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def collate_fn(self, batch_data):
        batch_input_ids = torch.stack([data['input_ids'] for data in batch_data])
        batch_size, seq_len = batch_input_ids.shape
        return {
            'input_ids': batch_input_ids[: , :-1].T.contiguous(),
            'target_ids': batch_input_ids[: , 1:].T.contiguous(),
            'position_idx': torch.arange(seq_len - 1, dtype=torch.long).unsqueeze(1).expand(-1, batch_size).contiguous(),
            'attention_mask': torch.tril(torch.ones((seq_len - 1, seq_len - 1), dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1).contiguous(),
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
