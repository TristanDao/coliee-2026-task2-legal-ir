import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict
from data import Task2Dataset

class Task2PairDataset(Dataset):
    def __init__(self, task2_dataset: Task2Dataset, tokenizer_name: str = "microsoft/deberta-v3-base", max_length: int = 512):
        """
        Wrapper around Task2Dataset to generate (query, paragraph) pairs for generic NLI/Entailment models.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = self._flatten_samples(task2_dataset)

    def _flatten_samples(self, dataset: Task2Dataset) -> List[Dict]:
        """
        Converts the hierarchical case structure into a flat list of (query, paragraph) pairs.
        """
        flat_samples = []
        for case in dataset:
            query = case["entailed_fragment"]
            # Optional: You might want to prepend base_case context here if needed
            # context = case["base_case"] 
            
            for para in case["paragraphs"]:
                flat_samples.append({
                    "case_id": case["case_id"],
                    "para_id": para["para_id"],
                    "query": query,
                    "paragraph": para["paragraph"],
                    "label": para["label"]
                })
        return flat_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        query = item["query"]
        paragraph = item["paragraph"]
        label = item["label"]

        # Tokenize pair
        inputs = self.tokenizer(
            query,
            paragraph,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "case_id": item["case_id"],
            "para_id": item["para_id"]
        }
