import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from data import Task2Dataset
from dataset import Task2PairDataset
from model import EntailmentModel
from scorer import get_scores
from evaluate import calculate_metrics, print_metrics
import os
import torch.nn.functional as F
from tqdm import tqdm

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=3, save_dir="model/best_model"):
    best_f1 = 0.0
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training Loop
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        print(f"Running Validation for Epoch {epoch+1}...")
        results = get_scores(model, val_dataloader, device)
        metrics = calculate_metrics(results, threshold=0.5)
        print_metrics(metrics)
        
        # Save best model
        current_f1 = metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"New best F1 score ({best_f1:.4f}). Saving model to {save_dir}...")
            model.save_pretrained(save_dir)


def main():
    # Configuration
    TRAIN_DATA_DIR = "Data/task2_train_files_2025"
    TRAIN_LABEL_PATH = "Data/task2_train_labels_2025.json"
    MODEL_NAME = "microsoft/deberta-v3-base"
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Loading data...")
    try:
        raw_dataset = Task2Dataset(data_dir=TRAIN_DATA_DIR, label_path=TRAIN_LABEL_PATH)
        print(f"Loaded {len(raw_dataset)} cases for training/validation.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # 2. Train/Val Split (80/20)
    train_size = int(0.8 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    train_subset, val_subset = random_split(raw_dataset, [train_size, val_size])
    print(f"Split data into {len(train_subset)} training and {len(val_subset)} validation cases.")

    # 3. Prepare Pair Datasets
    # Dataloaders need the flattened pairs (query, paragraph) 
    train_pair_dataset = Task2PairDataset(train_subset, tokenizer_name=MODEL_NAME)
    val_pair_dataset = Task2PairDataset(val_subset, tokenizer_name=MODEL_NAME)
    
    train_dataloader = DataLoader(train_pair_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_pair_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    print("Initializing model...")
    model = EntailmentModel(model_name=MODEL_NAME)
    model.to(DEVICE)
    
    # 5. Optimizer and Scheduler setup
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), # 10% warmup
        num_training_steps=total_steps
    )
    
    # 6. Start Training
    train(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        device=DEVICE,
        num_epochs=EPOCHS
    )

if __name__ == "__main__":
    main()
