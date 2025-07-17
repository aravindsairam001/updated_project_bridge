#!/usr/bin/env python3
"""
Dataset Preparation Helper for Continual Learning

This script helps prepare new datasets in the correct format for continual learning.
It can:
1. Convert JSON annotations to masks using the same class mapping
2. Organize images and masks into the required folder structure
3. Validate the dataset before training
"""

import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import argparse

# Same class mapping as in json_mask_min_class.py
MINIMAL_LABEL_MAP = {
    'rust': 1,
    'alligator crack': 2,
    'washouts/concrete corrosion': 3,
    'cavity': 4,
    'hollowareas': 5,
    'spalling': 6,
    'rockpocket': 7,
    'exposed rebars': 8,
    'crack': 9,
    'weathering': 10,
    'efflorescence': 11
}

def convert_json_to_mask(json_path, out_mask_path):
    """Convert JSON annotation to mask using the minimal class mapping"""
    with open(json_path) as f:
        data = json.load(f)

    # Support both 'height'/'width' and 'size' dict
    if 'height' in data and 'width' in data:
        height = data['height']
        width = data['width']
    elif 'size' in data:
        height = data['size']['height']
        width = data['size']['width']
    else:
        raise ValueError("Cannot find image size in JSON.")

    mask = np.zeros((height, width), dtype=np.uint8)

    # Process annotations
    for obj in data.get('objects', []):
        label = obj['classTitle'].strip().lower()
        if label not in MINIMAL_LABEL_MAP:
            continue  # Skip unknown classes
            
        points = np.array(obj['points']['exterior'], dtype=np.int32)
        class_id = MINIMAL_LABEL_MAP[label]
        cv2.fillPoly(mask, [points], class_id)

    cv2.imwrite(out_mask_path, mask)
    return mask

def prepare_dataset_from_json(source_dir, target_dir, images_subdir="images", annotations_subdir="ann"):
    """
    Prepare dataset from JSON annotations
    
    Expected source structure:
    source_dir/
        images/          # or custom images_subdir
            image1.jpg
            image2.jpg
        ann/            # or custom annotations_subdir  
            image1.jpg.json
            image2.jpg.json
    
    Creates target structure:
    target_dir/
        images/
            image1.jpg
            image2.jpg
        masks/
            image1.jpg.png
            image2.jpg.png
    """
    source_images_dir = os.path.join(source_dir, images_subdir)
    source_ann_dir = os.path.join(source_dir, annotations_subdir)
    
    target_images_dir = os.path.join(target_dir, "images")
    target_masks_dir = os.path.join(target_dir, "masks")
    
    # Create target directories
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_masks_dir, exist_ok=True)
    
    if not os.path.exists(source_images_dir):
        raise ValueError(f"Source images directory not found: {source_images_dir}")
    if not os.path.exists(source_ann_dir):
        raise ValueError(f"Source annotations directory not found: {source_ann_dir}")
    
    # Get all images
    image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    converted_count = 0
    skipped_count = 0
    
    for img_file in tqdm(image_files, desc="Processing dataset"):
        # Source paths
        source_img_path = os.path.join(source_images_dir, img_file)
        json_file = img_file + ".json"
        source_json_path = os.path.join(source_ann_dir, json_file)
        
        # Target paths
        target_img_path = os.path.join(target_images_dir, img_file)
        mask_file = img_file + ".png"
        target_mask_path = os.path.join(target_masks_dir, mask_file)
        
        if not os.path.exists(source_json_path):
            print(f"Warning: No annotation found for {img_file}")
            skipped_count += 1
            continue
        
        try:
            # Copy image
            shutil.copy2(source_img_path, target_img_path)
            
            # Convert annotation to mask
            convert_json_to_mask(source_json_path, target_mask_path)
            converted_count += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            skipped_count += 1
            # Clean up partial files
            if os.path.exists(target_img_path):
                os.remove(target_img_path)
            if os.path.exists(target_mask_path):
                os.remove(target_mask_path)
    
    print(f"\nDataset preparation complete!")
    print(f"✅ Converted: {converted_count} samples")
    print(f"⚠️  Skipped: {skipped_count} samples")
    print(f"📁 Output directory: {target_dir}")
    
    return converted_count

def prepare_dataset_from_existing_masks(source_dir, target_dir, images_subdir="images", masks_subdir="masks"):
    """
    Prepare dataset from existing images and masks
    
    Expected source structure:
    source_dir/
        images/         # or custom images_subdir
            image1.jpg
            image2.jpg
        masks/          # or custom masks_subdir
            image1.png  # or image1.jpg.png
            image2.png
    """
    source_images_dir = os.path.join(source_dir, images_subdir)
    source_masks_dir = os.path.join(source_dir, masks_subdir)
    
    target_images_dir = os.path.join(target_dir, "images")
    target_masks_dir = os.path.join(target_dir, "masks")
    
    # Create target directories
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_masks_dir, exist_ok=True)
    
    if not os.path.exists(source_images_dir):
        raise ValueError(f"Source images directory not found: {source_images_dir}")
    if not os.path.exists(source_masks_dir):
        raise ValueError(f"Source masks directory not found: {source_masks_dir}")
    
    # Get all images
    image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    copied_count = 0
    skipped_count = 0
    
    for img_file in tqdm(image_files, desc="Copying dataset"):
        # Source paths
        source_img_path = os.path.join(source_images_dir, img_file)
        
        # Try different mask naming conventions
        mask_candidates = [
            img_file.replace('.jpg', '.png').replace('.jpeg', '.png'),  # image.png
            img_file + '.png',  # image.jpg.png
            os.path.splitext(img_file)[0] + '.png'  # image_name.png
        ]
        
        source_mask_path = None
        for mask_candidate in mask_candidates:
            candidate_path = os.path.join(source_masks_dir, mask_candidate)
            if os.path.exists(candidate_path):
                source_mask_path = candidate_path
                break
        
        if source_mask_path is None:
            print(f"Warning: No mask found for {img_file}")
            skipped_count += 1
            continue
        
        # Target paths
        target_img_path = os.path.join(target_images_dir, img_file)
        target_mask_path = os.path.join(target_masks_dir, img_file + ".png")
        
        try:
            # Copy image and mask
            shutil.copy2(source_img_path, target_img_path)
            shutil.copy2(source_mask_path, target_mask_path)
            copied_count += 1
            
        except Exception as e:
            print(f"Error copying {img_file}: {e}")
            skipped_count += 1
    
    print(f"\nDataset preparation complete!")
    print(f"✅ Copied: {copied_count} samples")
    print(f"⚠️  Skipped: {skipped_count} samples")
    print(f"📁 Output directory: {target_dir}")
    
    return copied_count

def validate_dataset(dataset_dir):
    """Validate that the dataset is properly formatted for continual learning"""
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not os.path.exists(masks_dir):
        print(f"❌ Masks directory not found: {masks_dir}")
        return False
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
    
    print(f"📊 Dataset validation for: {dataset_dir}")
    print(f"📁 Images found: {len(image_files)}")
    print(f"🎭 Masks found: {len(mask_files)}")
    
    # Check image-mask pairs
    valid_pairs = 0
    invalid_pairs = []
    
    for img_file in image_files:
        expected_mask = img_file + ".png"
        mask_path = os.path.join(masks_dir, expected_mask)
        
        if os.path.exists(mask_path):
            valid_pairs += 1
        else:
            invalid_pairs.append(img_file)
    
    print(f"✅ Valid image-mask pairs: {valid_pairs}")
    
    if invalid_pairs:
        print(f"❌ Images without masks ({len(invalid_pairs)}):")
        for img_file in invalid_pairs[:5]:  # Show first 5
            print(f"   - {img_file}")
        if len(invalid_pairs) > 5:
            print(f"   ... and {len(invalid_pairs) - 5} more")
    
    # Validate mask content
    class_counts = {}
    
    for mask_file in tqdm(mask_files[:10], desc="Validating mask content"):  # Sample first 10
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
            
        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print(f"\n🏷️  Classes found in sample masks:")
    for class_id in sorted(class_counts.keys()):
        class_name = "background" if class_id == 0 else f"class_{class_id}"
        print(f"   Class {class_id} ({class_name}): {class_counts[class_id]} occurrences")
    
    # Check for valid class range
    max_class = max(class_counts.keys()) if class_counts else 0
    if max_class > 11:
        print(f"⚠️  Warning: Found class ID {max_class} > 11. Expected range: 0-11")
    
    success = valid_pairs > 0 and len(invalid_pairs) == 0
    
    if success:
        print(f"\n✅ Dataset validation passed! Ready for continual learning.")
    else:
        print(f"\n❌ Dataset validation failed. Please fix the issues above.")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for continual learning")
    parser.add_argument("--source", required=True, help="Source dataset directory")
    parser.add_argument("--target", required=True, help="Target dataset directory")
    parser.add_argument("--mode", choices=["json", "masks", "validate"], required=True,
                       help="Mode: 'json' (convert from JSON), 'masks' (copy existing), 'validate' (check format)")
    parser.add_argument("--images-subdir", default="images", help="Images subdirectory name (default: images)")
    parser.add_argument("--annotations-subdir", default="ann", help="Annotations subdirectory name (default: ann)")
    parser.add_argument("--masks-subdir", default="masks", help="Masks subdirectory name (default: masks)")
    
    args = parser.parse_args()
    
    if args.mode == "validate":
        validate_dataset(args.source)
    elif args.mode == "json":
        count = prepare_dataset_from_json(
            args.source, args.target, 
            args.images_subdir, args.annotations_subdir
        )
        if count > 0:
            print(f"\n🔍 Validating prepared dataset...")
            validate_dataset(args.target)
    elif args.mode == "masks":
        count = prepare_dataset_from_existing_masks(
            args.source, args.target,
            args.images_subdir, args.masks_subdir
        )
        if count > 0:
            print(f"\n🔍 Validating prepared dataset...")
            validate_dataset(args.target)

if __name__ == "__main__":
    # Interactive mode if no args provided
    import sys
    if len(sys.argv) == 1:
        print("🔧 Dataset Preparation for Continual Learning")
        print("=" * 50)
        
        mode = input("Choose mode:\n1. Convert from JSON annotations\n2. Copy existing masks\n3. Validate dataset\nEnter choice (1/2/3): ").strip()
        
        if mode == "1":
            source = input("Enter source directory path (with images/ and ann/ folders): ").strip()
            target = input("Enter target directory path: ").strip()
            try:
                count = prepare_dataset_from_json(source, target)
                if count > 0:
                    print(f"\n🔍 Validating prepared dataset...")
                    validate_dataset(target)
            except Exception as e:
                print(f"Error: {e}")
                
        elif mode == "2":
            source = input("Enter source directory path (with images/ and masks/ folders): ").strip()
            target = input("Enter target directory path: ").strip()
            try:
                count = prepare_dataset_from_existing_masks(source, target)
                if count > 0:
                    print(f"\n🔍 Validating prepared dataset...")
                    validate_dataset(target)
            except Exception as e:
                print(f"Error: {e}")
                
        elif mode == "3":
            dataset_dir = input("Enter dataset directory path to validate: ").strip()
            try:
                validate_dataset(dataset_dir)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice!")
    else:
        main()
