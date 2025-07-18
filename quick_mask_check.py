#!/usr/bin/env python3
"""
Quick Mask Class Checker

Simple script to quickly check what classes are in your mask files.
"""

import os
import cv2
import numpy as np
from collections import Counter

def quick_check_masks(dataset_dir, max_files=50):
    """Quick check of mask classes"""
    
    mask_dir = os.path.join(dataset_dir, "masks")
    if not os.path.exists(mask_dir):
        print(f"Masks directory not found: {mask_dir}")
        return
    
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    if not mask_files:
        print("No mask files found!")
        return
    
    print(f"Checking {min(len(mask_files), max_files)} mask files...")
    
    all_classes = set()
    class_counts = Counter()
    
    for i, mask_file in enumerate(mask_files[:max_files]):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Could not load: {mask_file}")
            continue
        
        unique_classes = np.unique(mask)
        all_classes.update(unique_classes)
        
        for cls in unique_classes:
            class_counts[int(cls)] += 1
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} files...")
    
    print(f"\n=== RESULTS ===")
    print(f"Found classes: {sorted(all_classes)}")
    print(f"Expected range: [0, 11]")
    
    # Check for problems
    invalid_classes = [cls for cls in all_classes if cls < 0 or cls > 11]
    if invalid_classes:
        print(f"❌ INVALID CLASSES FOUND: {invalid_classes}")
        print("These will cause training errors!")
    else:
        print("✅ All classes are in valid range")
    
    print(f"\nClass frequency (in how many files):")
    for cls in sorted(class_counts.keys()):
        print(f"  Class {cls:2d}: {class_counts[cls]:3d} files")

if __name__ == "__main__":
    dataset_dir = input("Enter dataset directory: ").strip()
    quick_check_masks(dataset_dir)
