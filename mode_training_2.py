import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Vision Transformer Model
class ViTDeepfake(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_classes=1):
        super(ViTDeepfake, self).__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get the feature dimension
        if 'base' in model_name:
            feature_dim = 768
        elif 'large' in model_name:
            feature_dim = 1024
        elif 'small' in model_name:
            feature_dim = 384
        else:
            feature_dim = 768  # default
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features from ViT
        features = self.vit(x)
        
        # Classification
        output = self.classifier(features)
        return output

# Transforms - ViT typically uses 224x224
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=0.3),
        A.MedianBlur(blur_limit=5, p=0.2),
        A.GaussianBlur(blur_limit=5, p=0.3),
    ], p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.0625, 
        scale_limit=0.15, 
        rotate_limit=20, 
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    ], p=0.2),
    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.3),
    A.CoarseDropout(
        max_holes=8, 
        max_height=32, 
        max_width=32, 
        min_holes=3,
        fill_value=0,
        p=0.3
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Training function with warmup
def train_model(model, train_loader, val_loader, epochs=50, device='cuda', 
                warmup_epochs=5, initial_lr=1e-5, max_lr=3e-4):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Warmup + Cosine annealing scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=max_lr/initial_lr, 
        total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs-warmup_epochs, 
        eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                labels_unsqueezed = labels.unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels_unsqueezed)
                
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()
        
        val_acc = accuracy_score(all_labels, all_predictions) * 100
        val_f1 = f1_score(all_labels, all_predictions)
        avg_val_loss = val_loss / len(val_loader)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(current_lr)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, 'vit_base_224_best.pth')
            print(f'✓ Model saved! Best Val Acc: {best_val_acc:.2f}%, Best Val F1: {best_val_f1:.4f}')
        
        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        print('-' * 70)
    
    # Save training history
    import json
    with open('vit_training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f'\nTraining completed!')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    print(f'Best Validation F1-Score: {best_val_f1:.4f}')
    
    return model, history

# Evaluation function
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            
            predictions = (outputs > 0.5).float().cpu().numpy()
            probs = outputs.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "="*50)
    print("Vision Transformer Results (224×224)")
    print("="*50)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("="*50)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# Data loading function
def load_data_from_folders(split='train', base_path='dataset'):
    """
    Load data from folder structure:
    dataset/
        train/
            real/
            fake/
        val/
            real/
            fake/
        test/
            real/
            fake/
    
    Returns:
        image_paths: list of image file paths
        labels: list of labels (0=real, 1=fake)
    """
    import os
    from glob import glob
    
    real_path = os.path.join(base_path, split, 'real')
    fake_path = os.path.join(base_path, split, 'fake')
    
    # Get all images (supporting common formats)
    real_images = []
    fake_images = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        real_images.extend(glob(os.path.join(real_path, ext)))
        real_images.extend(glob(os.path.join(real_path, ext.upper())))
        fake_images.extend(glob(os.path.join(fake_path, ext)))
        fake_images.extend(glob(os.path.join(fake_path, ext.upper())))
    
    image_paths = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)
    
    print(f"Loaded {split} data: {len(real_images)} real, {len(fake_images)} fake images")
    
    return image_paths, labels

# Main
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # Hyperparameters
    BATCH_SIZE = 32  # Reduce if OOM
    EPOCHS = 50
    NUM_WORKERS = 4
    
    # Load data
    print("\nLoading data...")
    train_paths, train_labels = load_data_from_folders('train')
    val_paths, val_labels = load_data_from_folders('val')
    test_paths, test_labels = load_data_from_folders('test')
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, val_transform)
    test_dataset = DeepfakeDataset(test_paths, test_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Initialize model
    print("\nInitializing Vision Transformer...")
    model = ViTDeepfake(
        model_name='vit_base_patch16_224',  # Options: vit_small, vit_base, vit_large
        pretrained=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=EPOCHS, 
        device=device,
        warmup_epochs=5,
        initial_lr=1e-5,
        max_lr=3e-4
    )
    
    # Load best model and evaluate
    print("\n" + "="*70)
    print("Evaluating on Test Set...")
    print("="*70)
    checkpoint = torch.load('vit_base_224_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    results = evaluate_model(model, test_loader, device=device)
    
    # Save results
    import json
    with open('vit_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✓ Training and evaluation completed!")
    print(f"✓ Best model saved as: vit_base_224_best.pth")
    print(f"✓ Training history saved as: vit_training_history.json")
    print(f"✓ Test results saved as: vit_test_results.json")