from pathlib import Path
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class EntailmentModel(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", num_labels: int = 2):
        super().__init__()

        project_root = Path(__file__).resolve().parent.parent
        cache_path = project_root / "model" / "cache"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=str(cache_path)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path):
        instance = cls(model_name=path)
        return instance
