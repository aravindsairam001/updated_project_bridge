import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import signal
import sys
import logging
import shutil
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continual_learning.log'),
        logging.StreamHandler()
    ]
)

def handle_kill_signal(signum, frame):
    reason = f"Continual learning interrupted or killed (signal {signum})"
    print(f"\n[!] {reason}")
    logging.info(reason)
    sys.exit(1)

signal.signal(signal.SIGINT, handle_kill_signal)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_kill_signal)  # kill command

# -------- CONFIG --------
NUM_CLASSES = 12  # Fixed: 11 defect classes + background
BATCH_SIZE = 2    # Smaller batch for continual learning
NUM_EPOCHS = 5    # Fewer epochs to prevent overfitting
IMAGE_SIZE = 384  # Reduced size for memory efficiency
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Continual Learning Parameters
INITIAL_LR = 1e-5      # Lower learning rate for fine-tuning
KNOWLEDGE_DISTILL_ALPHA = 0.7  # Weight for knowledge distillation
TEMPERATURE = 4         # Temperature for soft targets
EWC_LAMBDA = 400       # Elastic Weight Consolidation regularization
REPLAY_BUFFER_SIZE = 200  # Number of old samples to replay

# Model architecture (should match your trained model)
ARCHITECTURE = 'unetplusplus'  # Change this to match your model
ENCODER_NAME = 'efficientnet-b5'  # Change this to match your model
ENCODER_WEIGHTS = 'imagenet'

class ContinualDataset(Dataset):
    """Dataset class for continual learning"""
    def __init__(self, image_dir, mask_dir, transform=None, replay_samples=None):
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.transform = transform
        self.replay_samples = replay_samples or []
        
        # Get all new images and filter to only those with corresponding masks
        self.valid_pairs = []
        
        # Only scan directory if image_dir is provided and exists
        if image_dir and os.path.exists(image_dir):
            all_images = sorted(os.listdir(image_dir))
            
            for img_file in all_images:
                img_path = os.path.join(image_dir, img_file)
                # Handle .jpg, .png, and .jpg.png mask naming conventions
                if img_file.endswith(".jpg"):
                    mask_file = img_file + ".png"
                    if not os.path.exists(os.path.join(mask_dir, mask_file)):
                        mask_file = img_file.replace(".jpg", ".png")
                        if not os.path.exists(os.path.join(mask_dir, mask_file)):
                            mask_file = img_file.replace(".jpg", ".jpg.png")
                            if not os.path.exists(os.path.join(mask_dir, mask_file)):
                                mask_file = img_file.replace(".jpg", ".jpg")  # fallback, may not exist
                elif img_file.endswith(".png"):
                    mask_file = img_file
                    if not os.path.exists(os.path.join(mask_dir, mask_file)):
                        mask_file = img_file.replace(".png", ".jpg")
                else:
                    mask_file = img_file + ".png"
                    if not os.path.exists(os.path.join(mask_dir, mask_file)):
                        mask_file = img_file + ".jpg"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.valid_pairs.append(('new', img_file, mask_file))
                else:
                    print(f"Warning: Missing mask for {img_file}, skipping...")
        
        # Add replay samples
        self.valid_pairs.extend(self.replay_samples)
        
        new_samples_count = len(self.valid_pairs) - len(self.replay_samples)
        print(f"Found {len(self.valid_pairs)} samples ({new_samples_count} new + {len(self.replay_samples)} replay)")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        sample_type, img_file, mask_file = self.valid_pairs[idx]
        
        if sample_type == 'new':
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
        else:  # replay sample
            img_path = img_file  # Full path for replay samples
            mask_path = mask_file  # Full path for replay samples

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Validate and clip mask values to ensure they're in valid range [0, NUM_CLASSES-1]
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        
        # Check for any remaining invalid values and warn
        unique_values = np.unique(mask)
        if len(unique_values) > NUM_CLASSES:
            print(f"Warning: Mask {mask_path} has {len(unique_values)} unique values, expected max {NUM_CLASSES}")
            print(f"Unique values: {unique_values}")
            # Additional clipping to be safe
            mask = np.where(mask >= NUM_CLASSES, 0, mask)  # Set invalid values to background

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long(), sample_type

def get_continual_transform():
    """Conservative transforms for continual learning"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        
        # Light augmentations to prevent overfitting
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=10, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        
        # Quality enhancement
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform():
    """Validation transforms"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def create_model(architecture, encoder_name, num_classes):
    """Create model based on architecture choice"""
    if architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

def knowledge_distillation_loss(student_outputs, teacher_outputs, targets, alpha, temperature):
    """Knowledge distillation loss combining hard and soft targets"""
    # Hard target loss (standard cross-entropy)
    hard_loss = nn.CrossEntropyLoss()(student_outputs, targets)
    
    # Soft target loss (knowledge distillation)
    soft_targets = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    soft_student = nn.functional.log_softmax(student_outputs / temperature, dim=1)
    soft_loss = nn.functional.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Combine losses
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss

def compute_fisher_information(model, dataloader, num_samples=200):
    """Compute Fisher Information Matrix for EWC"""
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    
    model.eval()
    sample_count = 0
    
    with torch.enable_grad():
        for batch_idx, (images, masks, _) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, masks)
            
            model.zero_grad()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            sample_count += images.size(0)
    
    # Normalize by number of samples
    for name in fisher:
        fisher[name] /= sample_count
    
    return fisher

def ewc_loss(model, fisher, old_params, lambda_ewc):
    """Elastic Weight Consolidation loss"""
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - old_params[name]) ** 2).sum()
    return lambda_ewc * loss

def validate_dataset_masks(image_dir, mask_dir, num_classes=12):
    """Validate that all mask files have valid class labels"""
    print("Validating mask files...")
    
    if not image_dir or not os.path.exists(image_dir):
        return  # Skip validation for replay-only datasets
    
    all_images = sorted(os.listdir(image_dir))
    invalid_masks = []
    
    for img_file in all_images[:10]:  # Check first 10 files as sample
        # Find corresponding mask
        if img_file.endswith(".jpg"):
            mask_file = img_file + ".png"
            if not os.path.exists(os.path.join(mask_dir, mask_file)):
                mask_file = img_file.replace(".jpg", ".png")
                if not os.path.exists(os.path.join(mask_dir, mask_file)):
                    mask_file = img_file.replace(".jpg", ".jpg.png")
                    if not os.path.exists(os.path.join(mask_dir, mask_file)):
                        mask_file = img_file.replace(".jpg", ".jpg")
        elif img_file.endswith(".png"):
            mask_file = img_file
            if not os.path.exists(os.path.join(mask_dir, mask_file)):
                mask_file = img_file.replace(".png", ".jpg")
        else:
            mask_file = img_file + ".png"
            if not os.path.exists(os.path.join(mask_dir, mask_file)):
                mask_file = img_file + ".jpg"
        
        mask_path = os.path.join(mask_dir, mask_file)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_values = np.unique(mask)
                max_val = np.max(unique_values)
                if max_val >= num_classes:
                    invalid_masks.append((mask_file, unique_values, max_val))
    
    if invalid_masks:
        print(f"⚠️  Found {len(invalid_masks)} masks with invalid class labels:")
        for mask_file, unique_vals, max_val in invalid_masks:
            print(f"  - {mask_file}: max value {max_val}, unique: {unique_vals}")
        print(f"  Valid range is [0, {num_classes-1}]. Values will be clipped during training.")
    else:
        print("✅ All sampled masks have valid class labels.")

def save_replay_buffer(original_dataset_dir, replay_dir, num_samples):
    """Save samples from original dataset for replay"""
    os.makedirs(replay_dir, exist_ok=True)
    
    # Get original samples
    train_img_dir = os.path.join(original_dataset_dir, "train", "images")
    train_mask_dir = os.path.join(original_dataset_dir, "train", "masks")
    
    if not os.path.exists(train_img_dir):
        print(f"Warning: Original dataset not found at {train_img_dir}")
        return []
    
    all_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    
    # Randomly sample images
    import random
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    replay_samples = []
    for img_file in selected_images:
        img_path = os.path.join(train_img_dir, img_file)
        mask_file = img_file + ".png"
        mask_path = os.path.join(train_mask_dir, mask_file)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            # Copy to replay buffer
            replay_img_path = os.path.join(replay_dir, img_file)
            replay_mask_path = os.path.join(replay_dir, mask_file)
            
            shutil.copy2(img_path, replay_img_path)
            shutil.copy2(mask_path, replay_mask_path)
            
            replay_samples.append(('replay', replay_img_path, replay_mask_path))
    
    print(f"Saved {len(replay_samples)} samples to replay buffer")
    return replay_samples

def continual_train_epoch(model, teacher_model, dataloader, optimizer, fisher=None, old_params=None):
    """Train one epoch with continual learning"""
    model.train()
    teacher_model.eval()
    
    total_loss = 0
    total_kd_loss = 0
    total_ewc_loss = 0
    num_batches = len(dataloader)
    
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(mode='multiclass', from_logits=True)
    
    loop = tqdm(dataloader, desc="Continual Training", leave=False)
    for batch_idx, (images, masks, sample_types) in enumerate(loop):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        
        # Student predictions
        student_outputs = model(images)
        
        # Teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher_model(images)
        
        # Base losses
        ce = ce_loss(student_outputs, masks)
        dice = dice_loss(student_outputs, masks)
        base_loss = ce + dice
        
        # Knowledge distillation loss
        kd_loss = knowledge_distillation_loss(
            student_outputs, teacher_outputs, masks, 
            KNOWLEDGE_DISTILL_ALPHA, TEMPERATURE
        )
        
        # EWC loss (only if we have Fisher information)
        ewc_penalty = 0
        if fisher is not None and old_params is not None:
            ewc_penalty = ewc_loss(model, fisher, old_params, EWC_LAMBDA)
        
        # Total loss
        total_batch_loss = kd_loss + ewc_penalty
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_kd_loss += kd_loss.item()
        total_ewc_loss += ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else ewc_penalty
        
        # Update progress bar
        loop.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'kd': f'{kd_loss.item():.4f}',
            'ewc': f'{ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else ewc_penalty:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
        
        # Clear cache periodically
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'total_loss': total_loss / num_batches,
        'kd_loss': total_kd_loss / num_batches,
        'ewc_loss': total_ewc_loss / num_batches
    }

def validate_model(model, dataloader):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(mode='multiclass', from_logits=True)
    
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, (images, masks, _) in enumerate(loop):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            loss = ce_loss(outputs, masks) + dice_loss(outputs, masks)
            total_loss += loss.item()
            
            loop.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'avg_val_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    return total_loss / num_batches

def main():
    print(f"\n{'='*60}")
    print(f"{'CONTINUAL LEARNING FOR BRIDGE DEFECT DETECTION':^60}")
    print(f"{'='*60}")
    
    # Get paths from user
    pretrained_model_path = input("Enter path to your pretrained model (.pth file): ").strip()
    if not os.path.exists(pretrained_model_path):
        print(f"Error: Model file '{pretrained_model_path}' not found.")
        return
    
    new_data_dir = input("Enter path to new dataset directory (should contain images/ and masks/ folders): ").strip()
    if not os.path.exists(new_data_dir):
        print(f"Error: Dataset directory '{new_data_dir}' not found.")
        return
    
    original_dataset_dir = input("Enter path to original dataset directory (for replay buffer): ").strip()
    
    # Setup paths
    new_img_dir = os.path.join(new_data_dir, "images")
    new_mask_dir = os.path.join(new_data_dir, "masks")
    
    if not os.path.exists(new_img_dir) or not os.path.exists(new_mask_dir):
        print(f"Error: New dataset should have 'images' and 'masks' subdirectories.")
        return
    
    # Validate mask files before training
    validate_dataset_masks(new_img_dir, new_mask_dir, NUM_CLASSES)
    
    # Create replay buffer
    replay_dir = os.path.join(new_data_dir, "replay_buffer")
    replay_samples = save_replay_buffer(original_dataset_dir, replay_dir, REPLAY_BUFFER_SIZE)
    
    # Load teacher model (frozen copy of original)
    print("Loading teacher model...")
    teacher_model = create_model(ARCHITECTURE, ENCODER_NAME, NUM_CLASSES).to(DEVICE)
    teacher_model.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE))
    teacher_model.eval()
    
    # Load student model (will be updated)
    print("Loading student model...")
    student_model = create_model(ARCHITECTURE, ENCODER_NAME, NUM_CLASSES).to(DEVICE)
    student_model.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE))
    
    # Store old parameters for EWC
    old_params = {}
    for name, param in student_model.named_parameters():
        old_params[name] = param.data.clone()
    
    # Create datasets
    print("Creating datasets...")
    train_ds = ContinualDataset(
        new_img_dir, new_mask_dir, 
        transform=get_continual_transform(),
        replay_samples=replay_samples
    )
    
    # Create a small validation set from the new data (20% split)
    total_new_samples = len(os.listdir(new_img_dir))
    val_split = max(1, total_new_samples // 5)  # At least 1 sample for validation
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=False,
        drop_last=False
    )
    
    # Compute Fisher Information Matrix on original data (for EWC)
    print("Computing Fisher Information Matrix...")
    fisher = None
    if replay_samples:
        # Create a dataset with only replay samples for Fisher computation
        replay_dataset = ContinualDataset("", "", transform=get_val_transform(), replay_samples=replay_samples)
        replay_loader = DataLoader(replay_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        fisher = compute_fisher_information(teacher_model, replay_loader)
    
    # Setup optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        student_model.parameters(), 
        lr=INITIAL_LR, 
        weight_decay=1e-5  # Lower weight decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=INITIAL_LR/10
    )
    
    # Training loop
    print(f"\n[INFO] Starting continual learning...")
    print(f"[INFO] Architecture: {ARCHITECTURE} + {ENCODER_NAME}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] New samples: {total_new_samples}")
    print(f"[INFO] Replay samples: {len(replay_samples)}")
    print(f"[INFO] Epochs: {NUM_EPOCHS}")
    print(f"[INFO] Initial LR: {INITIAL_LR}")
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 40)
        
        # Training
        train_metrics = continual_train_epoch(
            student_model, teacher_model, train_loader, optimizer, fisher, old_params
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f} | "
              f"KD Loss: {train_metrics['kd_loss']:.4f} | "
              f"EWC Loss: {train_metrics['ewc_loss']:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = train_metrics['total_loss'] < best_loss
        if is_best:
            best_loss = train_metrics['total_loss']
            
            # Create model save path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f"dacl10k_{ARCHITECTURE}_{ENCODER_NAME.replace('-', '_')}_continual_{timestamp}.pth"
            
            torch.save(student_model.state_dict(), model_save_path)
            print(f"[✓] New best model saved! Loss: {train_metrics['total_loss']:.4f}")
            print(f"[✓] Model saved as: {model_save_path}")
        
        # Log training metrics
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['total_loss'],
            'kd_loss': train_metrics['kd_loss'],
            'ewc_loss': train_metrics['ewc_loss'],
            'lr': current_lr
        })
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save training history
    history_path = f"continual_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"{'CONTINUAL LEARNING COMPLETED':^60}")
    print(f"{'='*60}")
    print(f"Best training loss: {best_loss:.4f}")
    print(f"Training history saved: {history_path}")
    print(f"Final model: {model_save_path}")
    
    # Cleanup replay buffer if desired
    cleanup = input("\nDelete replay buffer? (y/N): ").strip().lower()
    if cleanup == 'y':
        shutil.rmtree(replay_dir)
        print("Replay buffer deleted.")

if __name__ == "__main__":
    main()
