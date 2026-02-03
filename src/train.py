import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import EmotionDataLoader
from src.model import DualHeadEmotionClassifier, MultiTaskLoss

class EmotionDataset(Dataset):
    def __init__(self, texts, primary_labels, secondary_labels, tokenizer, max_length=128):
        self.texts = texts
        self.primary_labels = primary_labels
        self.secondary_labels = secondary_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'primary_label': self.primary_labels[idx],
            'secondary_label': self.secondary_labels[idx]
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    primary_labels = torch.tensor([item['primary_label'] for item in batch])
    secondary_labels = torch.tensor([item['secondary_label'] for item in batch])
    return input_ids, attention_mask, primary_labels, secondary_labels

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    all_primary_preds = []
    all_primary_labels = []
    all_secondary_preds = []
    all_secondary_labels = []
    
    for input_ids, attention_mask, primary_labels, secondary_labels in tqdm(dataloader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        primary_labels = primary_labels.to(device)
        secondary_labels = secondary_labels.to(device)
        
        optimizer.zero_grad()
        primary_logits, secondary_logits = model(input_ids, attention_mask)
        
        loss, _, _ = criterion(primary_logits, secondary_logits, primary_labels, secondary_labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
        all_primary_preds.extend(primary_logits.argmax(dim=-1).cpu().detach().numpy())
        all_primary_labels.extend(primary_labels.cpu().numpy())
        all_secondary_preds.extend(secondary_logits.argmax(dim=-1).cpu().detach().numpy())
        all_secondary_labels.extend(secondary_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    primary_acc = accuracy_score(all_primary_labels, all_primary_preds)
    secondary_acc = accuracy_score(all_secondary_labels, all_secondary_preds)
    primary_f1 = f1_score(all_primary_labels, all_primary_preds, average='weighted', zero_division=0)
    secondary_f1 = f1_score(all_secondary_labels, all_secondary_preds, average='weighted', zero_division=0)
    
    return {
        'loss': avg_loss,
        'primary_acc': primary_acc,
        'secondary_acc': secondary_acc,
        'primary_f1': primary_f1,
        'secondary_f1': secondary_f1
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_primary_preds = []
    all_primary_labels = []
    all_secondary_preds = []
    all_secondary_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, primary_labels, secondary_labels in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            primary_labels = primary_labels.to(device)
            secondary_labels = secondary_labels.to(device)
            
            primary_logits, secondary_logits = model(input_ids, attention_mask)
            
            loss, _, _ = criterion(primary_logits, secondary_logits, primary_labels, secondary_labels)
            total_loss += loss.item()
            
            all_primary_preds.extend(primary_logits.argmax(dim=-1).cpu().numpy())
            all_primary_labels.extend(primary_labels.cpu().numpy())
            all_secondary_preds.extend(secondary_logits.argmax(dim=-1).cpu().numpy())
            all_secondary_labels.extend(secondary_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    primary_acc = accuracy_score(all_primary_labels, all_primary_preds)
    secondary_acc = accuracy_score(all_secondary_labels, all_secondary_preds)
    primary_f1 = f1_score(all_primary_labels, all_primary_preds, average='weighted', zero_division=0)
    secondary_f1 = f1_score(all_secondary_labels, all_secondary_preds, average='weighted', zero_division=0)
    primary_macro_f1 = f1_score(all_primary_labels, all_primary_preds, average='macro', zero_division=0)
    secondary_macro_f1 = f1_score(all_secondary_labels, all_secondary_preds, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'primary_acc': primary_acc,
        'secondary_acc': secondary_acc,
        'primary_f1': primary_f1,
        'secondary_f1': secondary_f1,
        'primary_macro_f1': primary_macro_f1,
        'secondary_macro_f1': secondary_macro_f1,
        'primary_preds': all_primary_preds,
        'primary_labels': all_primary_labels,
        'secondary_preds': all_secondary_preds,
        'secondary_labels': all_secondary_labels
    }


def sanity_check(model, dataloader, criterion, optimizer, device, num_batches=3):
    print("\n" + "="*50)
    print("SANITY CHECK: Overfitting on small batch")
    print("="*50)
    
    model.train()
    batch_data = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        batch_data.append(batch)
    
    for epoch in range(20):
        total_loss = 0
        correct_primary = 0
        correct_secondary = 0
        total = 0
        
        for input_ids, attention_mask, primary_labels, secondary_labels in batch_data:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            primary_labels = primary_labels.to(device)
            secondary_labels = secondary_labels.to(device)
            
            optimizer.zero_grad()
            primary_logits, secondary_logits = model(input_ids, attention_mask)
            loss, _, _ = criterion(primary_logits, secondary_logits, primary_labels, secondary_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct_primary += (primary_logits.argmax(dim=-1) == primary_labels).sum().item()
            correct_secondary += (secondary_logits.argmax(dim=-1) == secondary_labels).sum().item()
            total += primary_labels.size(0)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={total_loss/num_batches:.4f}, Primary Acc={correct_primary/total:.4f}, Secondary Acc={correct_secondary/total:.4f}")
    
    final_acc = (correct_primary + correct_secondary) / (2 * total)
    if final_acc > 0.8:
        print("✓ Sanity check PASSED: Model can overfit on small batch")
    else:
        print("✗ Sanity check FAILED: Model cannot overfit - check architecture")
    print("="*50 + "\n")
    return final_acc > 0.8


def main():
    DATA_PATH = "C:\\Users\\akash\\Downloads\\HACK"
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    EPOCHS = 50
    BATCH_SIZE = 16
    ENCODER_LR = 2e-5
    HEAD_LR = 1e-3
    WARMUP_EPOCHS = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    loader = EmotionDataLoader(
        primary_path=os.path.join(DATA_PATH, "dataset.csv.xlsx - Sheet1.csv"),
        secondary_path=os.path.join(DATA_PATH, "Secondary_Emotions.xlsx - Sheet1.csv")
    )
    
    df = loader.load_data()
    print(f"Total samples: {len(df)}")
    
    train_df, test_df = loader.get_train_test_split(df)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    primary_weights, secondary_weights = loader.compute_class_weights(train_df)
    label_info = loader.get_label_info()
    
    print(f"Primary emotions: {label_info['num_primary']}")
    print(f"Secondary emotions: {label_info['num_secondary']}")
    
    print("\nLabel distribution (Top 10):")
    primary_dist = train_df['Primary'].value_counts().head(10)
    print("Primary:", dict(primary_dist))
    
    with open(os.path.join(OUTPUT_PATH, "label_info.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'primary_labels': label_info['primary_labels'],
            'secondary_labels': label_info['secondary_labels'],
            'primary_label2id': label_info['primary_label2id'],
            'secondary_label2id': label_info['secondary_label2id'],
            'primary_id2label': {str(k): v for k, v in label_info['primary_id2label'].items()},
            'secondary_id2label': {str(k): v for k, v in label_info['secondary_id2label'].items()}
        }, f, ensure_ascii=False, indent=2)
    
    print("Initializing model...")
    model = DualHeadEmotionClassifier(
        num_primary_classes=label_info['num_primary'],
        num_secondary_classes=label_info['num_secondary']
    )
    model = model.to(device)
    
    train_dataset = EmotionDataset(
        train_df['Poem'].tolist(),
        train_df['primary_id'].tolist(),
        train_df['secondary_id'].tolist(),
        model.tokenizer
    )
    
    test_dataset = EmotionDataset(
        test_df['Poem'].tolist(),
        test_df['primary_id'].tolist(),
        test_df['secondary_id'].tolist(),
        model.tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    criterion = MultiTaskLoss(
        primary_class_weights=primary_weights,
        secondary_class_weights=secondary_weights
    ).to(device)
    
    print("\n--- Phase 1: Training heads only (encoder frozen) ---")
    model.freeze_encoder()
    head_optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=HEAD_LR,
        weight_decay=0.01
    )
    
    sanity_passed = sanity_check(model, train_loader, criterion, head_optimizer, device)
    
    model = DualHeadEmotionClassifier(
        num_primary_classes=label_info['num_primary'],
        num_secondary_classes=label_info['num_secondary']
    )
    model = model.to(device)
    model.freeze_encoder()
    
    head_optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=HEAD_LR,
        weight_decay=0.01
    )
    
    best_f1 = 0
    history = []
    
    print("\nStarting training...")
    for epoch in range(WARMUP_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS} (Heads only)")
        
        train_metrics = train_epoch(model, train_loader, criterion, head_optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Primary F1: {train_metrics['primary_f1']:.4f}, Secondary F1: {train_metrics['secondary_f1']:.4f}")
        
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Primary F1: {test_metrics['primary_f1']:.4f}, Secondary F1: {test_metrics['secondary_f1']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'phase': 'heads_only',
            'train': {k: v for k, v in train_metrics.items()},
            'test': {k: v for k, v in test_metrics.items() if k not in ['primary_preds', 'primary_labels', 'secondary_preds', 'secondary_labels']}
        })
    
    print("\n--- Phase 2: Fine-tuning full model (encoder unfrozen) ---")
    model.unfreeze_encoder()
    
    optimizer_params = model.get_optimizer_params(encoder_lr=ENCODER_LR, head_lr=HEAD_LR)
    full_optimizer = optim.AdamW(optimizer_params)
    
    # Cosine Annealing for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        full_optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    for epoch in range(WARMUP_EPOCHS, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS} (Full model)")
        
        train_metrics = train_epoch(model, train_loader, criterion, full_optimizer, device, scheduler)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Primary F1: {train_metrics['primary_f1']:.4f}, Secondary F1: {train_metrics['secondary_f1']:.4f}")
        
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Primary F1: {test_metrics['primary_f1']:.4f}, Secondary F1: {test_metrics['secondary_f1']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'phase': 'full_model',
            'train': {k: v for k, v in train_metrics.items()},
            'test': {k: v for k, v in test_metrics.items() if k not in ['primary_preds', 'primary_labels', 'secondary_preds', 'secondary_labels']}
        })
        
        avg_f1 = (test_metrics['primary_f1'] + test_metrics['secondary_f1']) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_info': label_info,
                'epoch': epoch + 1,
                'test_metrics': {k: v for k, v in test_metrics.items() if k not in ['primary_preds', 'primary_labels', 'secondary_preds', 'secondary_labels']}
            }, os.path.join(OUTPUT_PATH, "best_model.pt"))
            print(f"Saved best model with avg F1: {best_f1:.4f}")
    
    with open(os.path.join(OUTPUT_PATH, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Primary Emotion - Accuracy: {test_metrics['primary_acc']:.4f}, Weighted F1: {test_metrics['primary_f1']:.4f}, Macro F1: {test_metrics['primary_macro_f1']:.4f}")
    print(f"Secondary Emotion - Accuracy: {test_metrics['secondary_acc']:.4f}, Weighted F1: {test_metrics['secondary_f1']:.4f}, Macro F1: {test_metrics['secondary_macro_f1']:.4f}")
    
    np.save(os.path.join(OUTPUT_PATH, "primary_preds.npy"), np.array(test_metrics['primary_preds']))
    np.save(os.path.join(OUTPUT_PATH, "primary_labels.npy"), np.array(test_metrics['primary_labels']))
    np.save(os.path.join(OUTPUT_PATH, "secondary_preds.npy"), np.array(test_metrics['secondary_preds']))
    np.save(os.path.join(OUTPUT_PATH, "secondary_labels.npy"), np.array(test_metrics['secondary_labels']))
    
    print(f"\nModel and results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
