#!/usr/bin/env python3
"""
Mask Class Analyzer for Bridge Defect Detection

This script analyzes mask files in a dataset to:
1. Show all unique class labels found
2. Count pixels per class
3. Display class distribution
4. Identify potential issues (invalid classes, missing classes)
5. Show sample mask visualizations

Usage:
    python check_mask_classes.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse

# Expected class mapping (from your training setup)
EXPECTED_CLASSES = {
    0: "Background",
    1: "Rust", 
    2: "ACrack",
    3: "WConccor", 
    4: "Cavity",
    5: "Hollowareas",
    6: "Spalling",
    7: "Rockpocket",
    8: "ExposedRebars",
    9: "Crack",
    10: "Weathering",
    11: "Efflorescence"
}

# Color map for visualization
CLASS_COLORS = {
    0: (0, 0, 0),           # Background - black
    1: (255, 0, 0),         # Rust - red
    2: (255, 255, 0),       # ACrack - yellow
    3: (255, 0, 255),       # WConccor - magenta
    4: (0, 255, 255),       # Cavity - cyan
    5: (255, 128, 0),       # Hollowareas - orange
    6: (0, 255, 128),       # Spalling - spring green
    7: (255, 0, 128),       # Rockpocket - rose
    8: (128, 255, 0),       # ExposedRebars - lime
    9: (0, 128, 255),       # Crack - azure
    10: (128, 255, 255),    # Weathering - light cyan
    11: (255, 128, 192),    # Efflorescence - light pink
}

def find_mask_files(dataset_dir, images_subdir="images", masks_subdir="masks"):
    """Find all mask files in the dataset"""
    img_dir = os.path.join(dataset_dir, images_subdir)
    mask_dir = os.path.join(dataset_dir, masks_subdir)
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Error: Could not find {img_dir} or {mask_dir}")
        return []
    
    # Get all image files
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    mask_files = []
    missing_masks = []
    
    for img_file in image_files:
        # Try different mask naming conventions
        possible_masks = []
        if img_file.lower().endswith('.jpg') or img_file.lower().endswith('.jpeg'):
            base_name = os.path.splitext(img_file)[0]
            possible_masks = [
                img_file + ".png",           # image.jpg.png
                base_name + ".png",         # image.png
                img_file + ".jpg.png",      # image.jpg.jpg.png
                base_name + ".jpg"          # image.jpg
            ]
        elif img_file.lower().endswith('.png'):
            base_name = os.path.splitext(img_file)[0]
            possible_masks = [
                img_file,                   # image.png
                base_name + ".jpg"          # image.jpg
            ]
        
        # Find first existing mask
        mask_found = False
        for mask_name in possible_masks:
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                mask_files.append(mask_path)
                mask_found = True
                break
        
        if not mask_found:
            missing_masks.append(img_file)
    
    if missing_masks:
        print(f"⚠️  Found {len(missing_masks)} images without corresponding masks")
        if len(missing_masks) <= 10:
            print("Missing masks for:", missing_masks)
        else:
            print("Missing masks for:", missing_masks[:10], "... and more")
    
    print(f"Found {len(mask_files)} mask files to analyze")
    return mask_files

def analyze_single_mask(mask_path):
    """Analyze a single mask file"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, f"Could not load {mask_path}"
    
    unique_values, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    class_stats = {}
    for value, count in zip(unique_values, counts):
        percentage = (count / total_pixels) * 100
        class_stats[int(value)] = {
            'pixels': int(count),
            'percentage': percentage
        }
    
    return class_stats, None

def create_class_visualization(mask_path, output_path=None):
    """Create a colorized visualization of a mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Create RGB visualization
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in np.unique(mask):
        if class_id in CLASS_COLORS:
            color = CLASS_COLORS[class_id]
        else:
            # Unknown class - use white
            color = (255, 255, 255)
        
        colored_mask[mask == class_id] = color
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    
    return colored_mask

def analyze_dataset_masks(dataset_dir, max_files=None, save_visualizations=False):
    """Analyze all masks in a dataset"""
    print(f"\n{'='*60}")
    print(f"{'MASK CLASS ANALYSIS':^60}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_dir}")
    
    # Find all mask files
    mask_files = find_mask_files(dataset_dir)
    
    if not mask_files:
        print("No mask files found!")
        return
    
    if max_files:
        mask_files = mask_files[:max_files]
        print(f"Analyzing first {len(mask_files)} files...")
    
    # Analyze all masks
    all_class_stats = defaultdict(lambda: {'pixels': 0, 'files': 0})
    file_stats = []
    errors = []
    
    print("\nAnalyzing masks...")
    for mask_path in tqdm(mask_files):
        class_stats, error = analyze_single_mask(mask_path)
        
        if error:
            errors.append(error)
            continue
        
        # Accumulate statistics
        file_info = {
            'path': mask_path,
            'classes': class_stats
        }
        file_stats.append(file_info)
        
        for class_id, stats in class_stats.items():
            all_class_stats[class_id]['pixels'] += stats['pixels']
            all_class_stats[class_id]['files'] += 1
    
    # Calculate total pixels across all files
    total_pixels = sum(stats['pixels'] for stats in all_class_stats.values())
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{'ANALYSIS RESULTS':^60}")
    print(f"{'='*60}")
    
    print(f"📊 Files analyzed: {len(file_stats)}")
    print(f"❌ Files with errors: {len(errors)}")
    print(f"🎯 Total pixels: {total_pixels:,}")
    
    # Show class distribution
    print(f"\n{'CLASS DISTRIBUTION':^60}")
    print("-" * 60)
    print(f"{'Class':<12} {'Name':<15} {'Files':<8} {'Pixels':<12} {'Percentage'}")
    print("-" * 60)
    
    found_classes = sorted(all_class_stats.keys())
    for class_id in found_classes:
        stats = all_class_stats[class_id]
        class_name = EXPECTED_CLASSES.get(class_id, "UNKNOWN")
        percentage = (stats['pixels'] / total_pixels) * 100 if total_pixels > 0 else 0
        
        status = "✅" if class_id in EXPECTED_CLASSES else "❌"
        print(f"{status} {class_id:<8} {class_name:<15} {stats['files']:<8} {stats['pixels']:<12,} {percentage:>6.2f}%")
    
    # Check for missing expected classes
    missing_classes = set(EXPECTED_CLASSES.keys()) - set(found_classes)
    if missing_classes:
        print(f"\n⚠️  Missing expected classes: {sorted(missing_classes)}")
        for class_id in sorted(missing_classes):
            print(f"   - Class {class_id}: {EXPECTED_CLASSES[class_id]}")
    
    # Check for unexpected classes
    unexpected_classes = set(found_classes) - set(EXPECTED_CLASSES.keys())
    if unexpected_classes:
        print(f"\n❌ Found unexpected classes: {sorted(unexpected_classes)}")
        print("   These classes will cause training errors!")
        
        # Show files with unexpected classes
        print("\n   Files with unexpected classes:")
        for file_info in file_stats[:10]:  # Show first 10
            file_classes = set(file_info['classes'].keys())
            if file_classes & unexpected_classes:
                unexpected_in_file = file_classes & unexpected_classes
                print(f"   - {os.path.basename(file_info['path'])}: classes {sorted(unexpected_in_file)}")
    
    # Show errors
    if errors:
        print(f"\n❌ Errors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
    
    # Create sample visualizations
    if save_visualizations and file_stats:
        print(f"\n📸 Creating sample visualizations...")
        vis_dir = os.path.join(dataset_dir, "mask_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize up to 5 sample masks
        for i, file_info in enumerate(file_stats[:5]):
            mask_path = file_info['path']
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            output_path = os.path.join(vis_dir, f"sample_{i+1}_{base_name}_colored.png")
            
            colored_mask = create_class_visualization(mask_path, output_path)
            if colored_mask is not None:
                print(f"   - Saved: {output_path}")
        
        # Create legend
        create_legend(vis_dir)
    
    # Summary recommendations
    print(f"\n{'RECOMMENDATIONS':^60}")
    print("-" * 60)
    
    if unexpected_classes:
        print("🔧 REQUIRED FIXES:")
        print("   1. Remove or remap unexpected classes before training")
        print("   2. Valid class range is [0, 11]")
        print("   3. Consider using mask preprocessing script")
    
    if missing_classes:
        print("ℹ️  INFO:")
        print("   - Some expected classes are missing (this is okay)")
        print("   - Model will still train on available classes")
    
    if not unexpected_classes and not errors:
        print("✅ Dataset looks good for training!")
        print("   - All classes are in valid range")
        print("   - No errors found in mask files")
    
    return {
        'class_stats': dict(all_class_stats),
        'file_stats': file_stats,
        'errors': errors,
        'found_classes': found_classes,
        'unexpected_classes': unexpected_classes,
        'missing_classes': missing_classes
    }

def create_legend(output_dir):
    """Create a color legend for class visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = 0
    for class_id, class_name in EXPECTED_CLASSES.items():
        color = [c/255.0 for c in CLASS_COLORS[class_id]]  # Convert to matplotlib format
        ax.barh(y_pos, 1, color=color, height=0.8)
        ax.text(1.1, y_pos, f"Class {class_id}: {class_name}", 
                va='center', fontsize=10)
        y_pos += 1
    
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, len(EXPECTED_CLASSES) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Class Color Legend", fontsize=14, fontweight='bold')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    legend_path = os.path.join(output_dir, "class_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   - Saved legend: {legend_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze mask classes in a dataset")
    parser.add_argument("dataset_dir", nargs="?", help="Path to dataset directory")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to analyze")
    parser.add_argument("--visualize", action="store_true", help="Create sample visualizations")
    parser.add_argument("--images-dir", default="images", help="Images subdirectory name")
    parser.add_argument("--masks-dir", default="masks", help="Masks subdirectory name")
    
    args = parser.parse_args()
    
    # Get dataset directory
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = input("Enter dataset directory path: ").strip()
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return
    
    # Analyze the dataset
    results = analyze_dataset_masks(
        dataset_dir, 
        max_files=args.max_files, 
        save_visualizations=args.visualize
    )
    
    # Interactive mode for detailed analysis
    while True:
        print(f"\n{'INTERACTIVE OPTIONS':^60}")
        print("-" * 60)
        print("1. Show detailed file statistics")
        print("2. Create sample visualizations")
        print("3. Show files with specific class")
        print("4. Export results to JSON")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            # Show detailed file stats
            print(f"\nDetailed file statistics (showing first 10 files):")
            for i, file_info in enumerate(results['file_stats'][:10]):
                print(f"\n{i+1}. {os.path.basename(file_info['path'])}")
                for class_id, stats in file_info['classes'].items():
                    class_name = EXPECTED_CLASSES.get(class_id, "UNKNOWN")
                    print(f"   Class {class_id} ({class_name}): {stats['pixels']:,} pixels ({stats['percentage']:.1f}%)")
        
        elif choice == "2":
            # Create visualizations
            vis_dir = os.path.join(dataset_dir, "mask_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            print("Creating visualizations...")
            for i, file_info in enumerate(results['file_stats'][:5]):
                mask_path = file_info['path']
                base_name = os.path.splitext(os.path.basename(mask_path))[0]
                output_path = os.path.join(vis_dir, f"sample_{i+1}_{base_name}_colored.png")
                
                colored_mask = create_class_visualization(mask_path, output_path)
                if colored_mask is not None:
                    print(f"Saved: {output_path}")
            
            create_legend(vis_dir)
            print(f"Visualizations saved to: {vis_dir}")
        
        elif choice == "3":
            # Show files with specific class
            try:
                target_class = int(input("Enter class ID to search for: "))
                print(f"\nFiles containing class {target_class}:")
                count = 0
                for file_info in results['file_stats']:
                    if target_class in file_info['classes']:
                        print(f"  - {os.path.basename(file_info['path'])}")
                        count += 1
                        if count >= 20:  # Limit output
                            print("  ... (showing first 20 matches)")
                            break
                print(f"Total files with class {target_class}: {count}")
            except ValueError:
                print("Invalid class ID")
        
        elif choice == "4":
            # Export to JSON
            import json
            output_file = os.path.join(dataset_dir, "mask_analysis.json")
            with open(output_file, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                export_data = {
                    'class_stats': {k: {'pixels': int(v['pixels']), 'files': int(v['files'])} 
                                   for k, v in results['class_stats'].items()},
                    'found_classes': [int(x) for x in results['found_classes']],
                    'unexpected_classes': [int(x) for x in results['unexpected_classes']],
                    'missing_classes': [int(x) for x in results['missing_classes']],
                    'total_files': len(results['file_stats']),
                    'error_count': len(results['errors'])
                }
                json.dump(export_data, f, indent=2)
            print(f"Results exported to: {output_file}")
        
        elif choice == "5":
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
