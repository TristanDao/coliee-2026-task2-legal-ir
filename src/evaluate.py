from typing import List, Dict
from collections import defaultdict

def calculate_metrics(results: List[Dict], threshold: float = 0.5):
    """
    Calculates Precision, Recall, and F1 based on prediction results.
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)
    """
    
    # Organize by case to handle context if needed (though global count is standard)
    # Here we accumulate global TP, FP, FN
    tp = 0
    fp = 0
    fn = 0
    
    # We also need to group by case_id because we need to know the full set of positives 
    # to calculate FN correctly if we were doing retrieval @ k, but here 
    # we have binary labels for every pair so we can just sum up.
    # WAIT: If there are entailment pairs that were NOT in the dataloader (missed candidates),
    # then FN would be wrong. But we assume the dataloader contains ALL candidates 
    # for the cases in the validation set.
    
    for item in results:
        score = item['score']
        label = item['label']
        
        if label is None:
            continue # Skip if no ground truth
            
        prediction = 1 if score >= threshold else 0
        
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 1:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def print_metrics(metrics: Dict):
    print("=" * 30)
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print("=" * 30)