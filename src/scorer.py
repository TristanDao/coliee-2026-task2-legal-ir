import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_scores(model, dataloader, device):
    """
    Runs inference on the dataloader and returns scores for entailment (class 1).
    Returns a list of dicts: {'case_id': str, 'para_id': str, 'score': float, 'label': int}
    """
    model.eval()
    model.to(device)
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get probability of entailment (class 1)
            entailment_scores = probs[:, 1].cpu().numpy()
            
            # Metadata might be lists of strings due to collate_fn
            case_ids = batch['case_id']
            para_ids = batch['para_id']
            labels = batch['labels'].cpu().numpy() if 'labels' in batch else [None] * len(case_ids)
            
            for i in range(len(case_ids)):
                results.append({
                    'case_id': case_ids[i],
                    'para_id': para_ids[i],
                    'score': float(entailment_scores[i]),
                    'label': int(labels[i]) if labels[i] is not None else None
                })
                
    return results
