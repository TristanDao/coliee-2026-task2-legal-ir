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
    TEST_DATA_DIR = "Data/task2_test_files_2025"
    TEST_LABEL_PATH = "Data/task2_test_labels_2025.json"
    MODEL_PATH = "model/best_model"
    BASE_MODEL_NAME = "microsoft/deberta-v3-base" # Needed for tokenizer
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path '{MODEL_PATH}' not found. Please run main.py first to train a model.")
        return

    # 1. Load Data
    print("Loading test data...")
    try:
        test_dataset = Task2Dataset(data_dir=TEST_DATA_DIR, label_path=TEST_LABEL_PATH)
        print(f"Loaded {len(test_dataset)} cases for testing.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # 2. Prepare Pair Dataset
    test_pair_dataset = Task2PairDataset(test_dataset, tokenizer_name=BASE_MODEL_NAME)
    test_dataloader = DataLoader(test_pair_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    print(f"Loading fine-tuned model from '{MODEL_PATH}'...")
    model = EntailmentModel(model_name=MODEL_PATH)
    model.to(DEVICE)
    
    # 4. Evaluation Loop
    print("Running evaluation on test set...")
    results = get_scores(model, test_dataloader, DEVICE)
    
    # 5. Calculate Metrics
    metrics = calculate_metrics(results, threshold=0.5)
    print_metrics(metrics)

if __name__ == "__main__":
    main()
