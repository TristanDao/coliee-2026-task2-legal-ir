import torch
from torch.utils.data import DataLoader
from data import Task2Dataset
from dataset import Task2PairDataset
from model import EntailmentModel
from scorer import get_scores
from evaluate import calculate_metrics, print_metrics
import os

def main():
    # Configuration
    DATA_DIR = "Data/task2_train_files_2025"
    LABEL_PATH = "Data/task2_train_labels_2025.json"
    MODEL_NAME = "microsoft/deberta-v3-base"
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Loading data...")
    try:
        raw_dataset = Task2Dataset(data_dir=DATA_DIR, label_path=LABEL_PATH)
        print(f"Loaded {len(raw_dataset)} cases.")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create dummy data for demonstration if file not found
        print("Creating dummy dataset for demonstration...")
        # Note: In real usage, ensure paths are correct.
        return

    # 2. Prepare Pair Dataset
    pair_dataset = Task2PairDataset(raw_dataset, tokenizer_name=MODEL_NAME)
    dataloader = DataLoader(pair_dataset, batch_size=BATCH_SIZE, shuffle=False) # Shuffle=False for evaluation
    
    # 3. Initialize Model
    print("Initializing model...")
    model = EntailmentModel(model_name=MODEL_NAME)
    model.to(DEVICE)
    
    # 4. Evaluation Loop (Zero-shot / Untrained)
    print("Running evaluation...")
    results = get_scores(model, dataloader, DEVICE)
    
    # 5. Calculate Metrics
    metrics = calculate_metrics(results, threshold=0.5)
    print_metrics(metrics)
    
    # 6. Save Logic (Optional)
    # model.save_pretrained("model/saved_model")

if __name__ == "__main__":
    main()
