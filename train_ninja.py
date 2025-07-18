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

# Set up logging
logging.basicConfig(filename='training_interrupt.log', level=logging.INFO, format='%(asctime)s %(message)s')

def handle_kill_signal(signum, frame):
    reason = f"Training interrupted or killed (signal {signum})"
    print(f"\n[!] {reason}")
    logging.info(reason)
    sys.exit(1)

signal.signal(signal.SIGINT, handle_kill_signal)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_kill_signal)  # kill command

# -------- CONFIG --------
# Optimized for training on internet bridge images
# Focus: Image quality enhancement rather than real-world simulation
NUM_CLASSES = 12  # 11 defect classes + background (matching minimal class subset)
BATCH_SIZE = 4    # Reduced for larger models (EfficientNet/ConvNeXt)
NUM_EPOCHS = 100   # Sufficient for convergence on internet images
IMAGE_SIZE = 512
DATASET_DIR = 'Datasets/dacl10k_ninja'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Architecture Options (choose one):
ARCHITECTURE = 'unetplusplus'  # Options: 'deeplabv3plus', 'unetplusplus', 'fpn', 'linknet', 'pspnet'
ENCODER_NAME = 'efficientnet-b5'  # Better options: 'efficientnet-b5', 'resnext101_32x8d', 'se_resnext101_32x4d'
ENCODER_WEIGHTS = 'imagenet'

MODEL_SAVE_PATH = f'dacl10k_{ARCHITECTURE}_{ENCODER_NAME.replace("-", "_")}_ver1.pth'

# Alternative high-performance configurations:
# ARCHITECTURE = 'fpn'; ENCODER_NAME = 'se_resnext101_32x4d'  # Best for detailed features
# ARCHITECTURE = 'unetplusplus'; ENCODER_NAME = 'efficientnet-b7'  # Best accuracy (if memory allows)
# ARCHITECTURE = 'deeplabv3plus'; ENCODER_NAME = 'resnext101_32x8d'  # Good balance
# --------------------------------

# -------- Dataset --------
class DACL10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all images and filter to only those with corresponding masks
        all_images = sorted(os.listdir(image_dir))
        self.valid_pairs = []
        
        for img_file in all_images:
            img_path = os.path.join(image_dir, img_file)
            # Handle both .png and .jpg.png mask naming conventions
            mask_file = img_file.replace(".jpg", ".png") if not img_file.endswith(".jpg.png") else img_file + ".png"
            if img_file.endswith(".jpg"):
                mask_file = img_file + ".png"  # For dacl10k_v2_train_XXXX.jpg -> dacl10k_v2_train_XXXX.jpg.png
            mask_path = os.path.join(mask_dir, mask_file)
            
            # Only include if both image and mask exist
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.valid_pairs.append((img_file, mask_file))
            else:
                print(f"Warning: Missing mask for {img_file}, skipping...")
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs out of {len(all_images)} total images")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.valid_pairs[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()

# -------- Transforms Optimized for Internet Images --------
def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        
        # Quality Enhancement for Internet Images
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.8),  # Enhance contrast and details
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),  # Sharpen blurry internet images
        
        # Basic Geometric Augmentations (conservative for internet images)
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),  # Small rotations only
        
        # Color and Lighting Adjustments (common in internet images)
        A.RandomBrightnessContrast(
            brightness_limit=0.3, 
            contrast_limit=0.3, 
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15, 
            sat_shift_limit=20, 
            val_shift_limit=15, 
            p=0.5
        ),
        
        # Handle Different Image Qualities from Internet
        A.OneOf([
            A.GaussNoise(noise_scale_factor=0.1, p=0.3),  # Handle noisy images
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # ISO noise
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),  # JPEG compression
        ], p=0.4),
        
        # Exposure and Gamma Corrections (common internet image issues)
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            A.ChannelShuffle(p=0.2),
        ], p=0.2),
        
        # Final preprocessing
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        
        # Quality enhancement for validation (consistent preprocessing)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Always enhance contrast
        
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# -------- Enhanced Training Function --------
def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(loop):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        
        # Update progress bar
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })

    return total_loss / num_batches

# -------- Enhanced Validation Function --------
def validate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, (images, masks) in enumerate(loop):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            # Update progress bar
            loop.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'avg_val_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    return total_loss / num_batches

# -------- Model Summary Function --------
def print_model_summary(model, architecture, encoder):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'MODEL SUMMARY':^50}")
    print("-" * 50)
    print(f"Architecture: {architecture}")
    print(f"Encoder: {encoder}")
    print(f"Input channels: 3")
    print(f"Output classes: {NUM_CLASSES}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    print("-" * 50)

# -------- Model Selection Function --------
def create_model(architecture, encoder_name, num_classes):
    """Create model based on architecture choice"""
    
    if architecture == 'unetplusplus':
        # UNet++ - Best for detailed boundary detection (excellent for cracks)
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'fpn':
        # Feature Pyramid Network - Great for multi-scale defects
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'deeplabv3plus':
        # DeepLabV3Plus - Good general performance
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'linknet':
        # LinkNet - Lightweight but effective
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'pspnet':
        # PSPNet - Good for contextual understanding
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

# -------- Architecture Recommendation Function --------
def recommend_architecture():
    """Recommend best architecture based on available resources and defect detection needs"""
    print(f"\n{'ARCHITECTURE RECOMMENDATIONS FOR BRIDGE DEFECT DETECTION':^70}")
    print("=" * 70)
    
    if torch.cuda.is_available():
        # gpu_memory = 8  # Assume 8GB GPU for this example
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"Available GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 12:
            print("🏆 RECOMMENDED: UNet++ with EfficientNet-B7 (Best Accuracy)")
            print("   - Excellent for fine crack detection")
            print("   - Superior boundary accuracy")
            print("   - High memory requirement but best performance")
            
        elif gpu_memory >= 8:
            print("🥇 RECOMMENDED: UNet++ with EfficientNet-B5 (Balanced)")
            print("   - Excellent crack and defect boundary detection")
            print("   - Good balance of accuracy and memory usage")
            print("   - Current configuration ✓")
            
        elif gpu_memory >= 6:
            print("🥈 RECOMMENDED: FPN with SE-ResNeXt101 (Multi-scale)")
            print("   - Great for various defect sizes")
            print("   - Efficient feature pyramid")
            print("   - Good for complex bridge structures")
            
        else:
            print("🥉 RECOMMENDED: LinkNet with EfficientNet-B3 (Lightweight)")
            print("   - Faster training and inference")
            print("   - Lower memory requirements")
            print("   - Still good performance for most defects")
    else:
        print("⚠️  CPU Mode: LinkNet with ResNet50 (Fastest)")
        print("   - Optimized for CPU inference")
        print("   - Reasonable training time")
    
    print("\nDEFECT-SPECIFIC RECOMMENDATIONS:")
    print("🔸 Cracks & Fine Details: UNet++ (Dense skip connections)")
    print("🔸 Multi-scale Defects: FPN (Feature pyramid)")
    print("🔸 Speed Priority: LinkNet (Lightweight)")
    print("🔸 General Purpose: DeepLabV3Plus (Proven performance)")
    print("=" * 70)

# -------- Main --------
def main():
    # Show architecture recommendations
    recommend_architecture()
    
    # Paths
    train_img_dir = os.path.join(DATASET_DIR, "train", "images")
    train_mask_dir = os.path.join(DATASET_DIR, "train", "masks")
    val_img_dir = os.path.join(DATASET_DIR, "val", "images")
    val_mask_dir = os.path.join(DATASET_DIR, "val", "masks")

    # Datasets & Loaders with better error handling
    print("Loading datasets...")
    
    try:
        train_ds = DACL10KDataset(train_img_dir, train_mask_dir, transform=get_train_transform())
        val_ds = DACL10KDataset(val_img_dir, val_mask_dir, transform=get_val_transform())
        
        print(f"[✓] Training samples: {len(train_ds)}")
        print(f"[✓] Validation samples: {len(val_ds)}")
        
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError("Empty dataset detected!")
        
    except Exception as e:
        print(f"[✗] Error loading datasets: {e}")
        print("Please check your dataset paths and structure.")
        return

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # Model with optimized architecture for bridge defect detection
    print(f"\n[INFO] Creating {ARCHITECTURE.upper()} model with {ENCODER_NAME} encoder...")
    
    model = create_model(ARCHITECTURE, ENCODER_NAME, NUM_CLASSES).to(DEVICE)
    
    print(f"[✓] Model created: {ARCHITECTURE.upper()} with {ENCODER_NAME} encoder")
    print(f"[✓] Training for {NUM_CLASSES} classes (11 defects + background)")
    print(f"[✓] Device: {DEVICE}")
    print(f"[✓] Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"[✓] Batch size: {BATCH_SIZE}")
    print(f"[✓] Epochs: {NUM_EPOCHS}")
    
    # Print model summary
    print_model_summary(model, ARCHITECTURE, ENCODER_NAME)

    # Advanced loss functions and optimizer for bridge defect detection
    ce_loss = nn.CrossEntropyLoss(weight=None)  # Can add class weights if needed
    dice_loss = DiceLoss(mode='multiclass', from_logits=True)
    
    def combined_loss(pred, target):
        ce = ce_loss(pred, target)
        dice = dice_loss(pred, target)
        return ce + dice
    
    # Better optimizer configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # More sophisticated learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.7, 
        patience=5, 
        min_lr=1e-6,
        cooldown=2
    )
    
    # Alternative: Cosine annealing with warm restarts
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
    # )

    # Enhanced training loop with better monitoring
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    print(f"\n{'='*60}")
    print(f"{'BRIDGE DEFECT DETECTION TRAINING':^60}")
    print(f"{'='*60}")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 40)
        
        # Training
        train_loss = train_one_epoch(model, train_loader, combined_loss, optimizer)
        
        # Validation
        val_loss = validate(model, val_loader, combined_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[✓] New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[!] No improvement. Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n[!] Early stopping triggered after {epoch+1} epochs")
            print(f"[✓] Best validation loss: {best_val_loss:.4f}")
            break
        
        # Log progress
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    print(f"\n{'='*60}")
    print(f"{'TRAINING COMPLETED':^60}")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved as: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()