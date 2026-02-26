import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer
from src.model import EntailmentModel

class InferencePipeline:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EntailmentModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, case_data: Dict[str, Any], threshold: float = 0.5) -> List[str]:
        """
        Predicts which paragraphs entail the query for a single case.
        Returns a list of para_ids.
        """
        query = case_data["entailed_fragment"]
        paragraphs = case_data["paragraphs"]
        
        inputs = []
        para_ids = []
        
        for para in paragraphs:
            para_ids.append(para["para_id"])
            inputs.append((query, para["paragraph"]))
            
        # Tokenize
        encoded = self.tokenizer(
            [p[0] for p in inputs], 
            [p[1] for p in inputs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()
            
        # Filter by threshold
        relevant_para_ids = []
        for pid, score in zip(para_ids, scores):
            if score >= threshold:
                relevant_para_ids.append(pid)
                
        return relevant_para_ids
