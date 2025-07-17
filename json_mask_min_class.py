import os
import json
import numpy as np
import cv2
from tqdm import tqdm


# Map class names to integer IDs - UPDATED TO MATCH ACTUAL DATASET
# These are the actual labels found in your JSON files
LABEL_MAP = {
    'alligator crack': 1,
    'bearing': 2,
    'cavity': 3,
    'crack': 4,
    'drainage': 5,
    'efflorescence': 6,
    'expansion joint': 7,
    'exposed rebars': 8,
    'graffiti': 9,
    'hollowareas': 10,
    'joint tape': 11,
    'protective equipment': 12,
    'restformwork': 13,
    'rockpocket': 14,
    'rust': 15,
    'spalling': 16,
    'washouts/concrete corrosion': 17,
    'weathering': 18,
    'wetspot': 19,
}

# Subset mapping for minimal classes (if you want to use only specific defects)
MINIMAL_LABEL_MAP = {
    'rust': 1,
    'alligator crack': 2,  # ACrack equivalent
    'washouts/concrete corrosion': 3,  # WConccor equivalent
    'cavity': 4,
    'hollowareas': 5,
    'spalling': 6,
    'rockpocket': 7,
    'exposed rebars': 8,
    'crack': 9,
    'weathering': 10,
    'efflorescence': 11
}

# Choose which mapping to use
USE_MINIMAL_SET = True  # Set to False to use all 19 classes
ACTIVE_LABEL_MAP = MINIMAL_LABEL_MAP if USE_MINIMAL_SET else LABEL_MAP

def convert_labelme_json(json_path, out_mask_path):
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

    # DACL10K format: 'objects', 'classTitle', 'points'->'exterior'
    for obj in data.get('objects', []):
        label = obj['classTitle'].strip().lower()  # Convert to lowercase for matching
        if label not in ACTIVE_LABEL_MAP:
            if not USE_MINIMAL_SET:  # Only show warning when using full set
                print(f"Warning: Label '{label}' not in LABEL_MAP, skipping.")
            continue
        points = np.array(obj['points']['exterior'], dtype=np.int32)
        class_id = ACTIVE_LABEL_MAP[label]
        cv2.fillPoly(mask, [points], class_id)

    cv2.imwrite(out_mask_path, mask)

def print_dataset_info():
    """Print information about the current dataset configuration"""
    print("=" * 60)
    print("DATASET CONFIGURATION")
    print("=" * 60)
    
    if USE_MINIMAL_SET:
        print("🎯 Using MINIMAL class set (11 defects + background)")
        print(f"📊 Total classes: {len(MINIMAL_LABEL_MAP) + 1}")
        print("🏷️  Included defects:")
        for label, class_id in MINIMAL_LABEL_MAP.items():
            print(f"   {class_id}: {label}")
    else:
        print("🎯 Using FULL class set (19 defects + background)")
        print(f"📊 Total classes: {len(LABEL_MAP) + 1}")
        print("🏷️  Included defects:")
        for label, class_id in LABEL_MAP.items():
            print(f"   {class_id}: {label}")
    
    print("\n🔧 For training, use:")
    print(f"   NUM_CLASSES = {len(ACTIVE_LABEL_MAP) + 1}")
    print(f"   ALLOWED_CLASS_IDS = set([{', '.join(map(str, range(1, len(ACTIVE_LABEL_MAP) + 1)))}])")
    print("=" * 60)

# Example usage
def main():
    # Get the JSON directory from user
    print_dataset_info()
    
    json_dir = input("\nEnter JSON directory path: ").strip()
    if not os.path.isdir(json_dir):
        print(f"Error: Directory '{json_dir}' does not exist.")
        return

    # Create output directory
    output_dir = f"{json_dir}_masks"
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in '{json_dir}'")
        return

    print(f"\nFound {len(json_files)} JSON files. Converting...")
    
    for json_file in tqdm(json_files, desc="Converting"):
        json_path = os.path.join(json_dir, json_file)
        mask_filename = json_file.replace('.json', '.png')
        mask_path = os.path.join(output_dir, mask_filename)
        
        try:
            convert_labelme_json(json_path, mask_path)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"\nConversion complete! Masks saved to: {output_dir}")
    print(f"Class configuration: {'MINIMAL' if USE_MINIMAL_SET else 'FULL'}")
    print(f"Total classes for training: {len(ACTIVE_LABEL_MAP) + 1}")

if __name__ == "__main__":
    main()