import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# DACL10K ninja dataset class mapping based on meta.json
# Map from class titles to integer IDs for mask generation
DACL10K_LABEL_MAP = {
    'alligator crack': 4,      # ACrack
    'bearing': 16,             # Bearing
    'cavity': 6,               # Cavity
    'crack': 12,               # Crack
    'drainage': 14,            # Drainage
    'efflorescence': 19,       # Efflorescence
    'expansion joint': 3,      # EJoint
    'exposed rebars': 11,      # ExposedRebars
    'graffiti': 17,            # Graffiti
    'hollowareas': 7,          # Hollowareas
    'joint tape': 8,           # JTape
    'protective equipment': 18, # PEquipment
    'restformwork': 13,        # Restformwork
    'rockpocket': 10,          # Rockpocket
    'rust': 2,                 # Rust
    'spalling': 9,             # Spalling
    'washouts/concrete corrosion': 5,  # WConccor
    'weathering': 15,          # Weathering
    'wetspot': 1,              # Wetspot
}

def convert_dacl10k_ninja_json(json_path, out_mask_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return False

    # Get image dimensions
    height = data['size']['height']
    width = data['size']['width']
    
    # Initialize mask with background (0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each object
    for obj in data['objects']:
        class_title = obj['classTitle'].lower().strip()
        
        # Map class title to ID
        if class_title not in DACL10K_LABEL_MAP:
            print(f"Warning: Class '{class_title}' not in DACL10K_LABEL_MAP, skipping.")
            continue
            
        class_id = DACL10K_LABEL_MAP[class_title]
        
        # Get polygon points
        if 'points' in obj and 'exterior' in obj['points']:
            points = np.array(obj['points']['exterior'], dtype=np.int32)
            
            # Fill polygon with class ID
            cv2.fillPoly(mask, [points], class_id)
        else:
            print(f"Warning: No valid points found for object in {json_path}")

    # Save mask
    try:
        cv2.imwrite(out_mask_path, mask)
        return True
    except Exception as e:
        print(f"Error saving mask to {out_mask_path}: {e}")
        return False

def convert_all_annotations(json_dir, mask_dir, split='train'):
    # Create output directory
    os.makedirs(mask_dir, exist_ok=True)
    
    # Find all JSON files
    if not os.path.exists(json_dir):
        print(f"Error: Directory {json_dir} does not exist!")
        return
        
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    successful_conversions = 0
    failed_conversions = 0
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc=f"Converting {split} annotations"):
        # Generate output filename
        base_name = os.path.splitext(json_file)[0]
        # Remove .jpg from the base name if it exists (since files are like "image.jpg.json")
        if base_name.endswith('.jpg'):
            base_name = base_name[:-4]
        
        json_path = os.path.join(json_dir, json_file)
        mask_path = os.path.join(mask_dir, f"{base_name}.png")
        
        # Convert annotation
        if convert_dacl10k_ninja_json(json_path, mask_path):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    print(f"\n[✓] Conversion complete!")
    print(f"    Successfully converted: {successful_conversions}")
    print(f"    Failed conversions: {failed_conversions}")
    print(f"    Masks saved to: {mask_dir}")

# def create_colorized_masks(mask_dir, color_mask_dir):
#     """
#     Create colorized versions of masks for visualization
    
#     Args:
#         mask_dir: Directory containing grayscale masks
#         color_mask_dir: Directory to save colorized masks
#     """
#     # Color map matching the one in inference_image.py
#     CLASS_COLORS = {
#         0: (0, 0, 0),           # 0: background (black)
#         1: (255, 0, 0),         # 1: Wetspot (bright red)
#         2: (0, 255, 0),         # 2: Rust (bright green)
#         3: (0, 0, 255),         # 3: EJoint (bright blue)
#         4: (255, 255, 0),       # 4: ACrack (yellow)
#         5: (255, 0, 255),       # 5: WConccor (magenta)
#         6: (0, 255, 255),       # 6: Cavity (cyan)
#         7: (255, 128, 0),       # 7: Hollowareas (orange)
#         8: (128, 0, 255),       # 8: JTape (purple)
#         9: (0, 255, 128),       # 9: Spalling (spring green)
#         10: (255, 0, 128),      # 10: Rockpocket (rose)
#         11: (128, 255, 0),      # 11: ExposedRebars (lime)
#         12: (0, 128, 255),      # 12: Crack (azure)
#         13: (255, 255, 128),    # 13: Restformwork (light yellow)
#         14: (255, 128, 255),    # 14: Drainage (pink)
#         15: (128, 255, 255),    # 15: Weathering (light cyan)
#         16: (255, 128, 128),    # 16: Bearing (light red)
#         17: (128, 255, 128),    # 17: Graffiti (light green)
#         18: (128, 128, 255),    # 18: PEquipment (light blue)
#         19: (255, 128, 192),    # 19: Efflorescence (light pink)
#     }
    
#     os.makedirs(color_mask_dir, exist_ok=True)
    
#     mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
#     for mask_file in tqdm(mask_files, desc="Creating colorized masks"):
#         mask_path = os.path.join(mask_dir, mask_file)
#         color_mask_path = os.path.join(color_mask_dir, mask_file.replace('.png', '_color.png'))
        
#         # Load grayscale mask
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             continue
            
#         # Create color mask
#         color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
#         for class_id, color in CLASS_COLORS.items():
#             color_mask[mask == class_id] = color
            
#         # Save colorized mask
#         cv2.imwrite(color_mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    
#     print(f"[✓] Colorized masks saved to: {color_mask_dir}")

if __name__ == "__main__":
    # Configuration
    base_dir = 'dacl10k_ninja'
    
    # Process training set
    train_json_dir = os.path.join(base_dir, 'train', 'ann')
    train_mask_dir = os.path.join(base_dir, 'masks_train')
    
    print("Converting training annotations...")
    convert_all_annotations(train_json_dir, train_mask_dir, 'train')
    
    # Process validation set
    val_json_dir = os.path.join(base_dir, 'val', 'ann')
    val_mask_dir = os.path.join(base_dir, 'masks_val')
    
    if os.path.exists(val_json_dir):
        print("\nConverting validation annotations...")
        convert_all_annotations(val_json_dir, val_mask_dir, 'val')
    else:
        print(f"Validation directory {val_json_dir} not found, skipping...")
    
    # Process test set
    test_json_dir = os.path.join(base_dir, 'test', 'ann')
    test_mask_dir = os.path.join(base_dir, 'masks_test')
    
    if os.path.exists(test_json_dir):
        print("\nConverting test annotations...")
        convert_all_annotations(test_json_dir, test_mask_dir, 'test')
    else:
        print(f"Test directory {test_json_dir} not found, skipping...")
    
    # # Create colorized versions for visualization
    # print("\nCreating colorized masks for visualization...")
    # create_colorized_masks(train_mask_dir, os.path.join(base_dir, 'masks_train_color'))
    
    # if os.path.exists(val_mask_dir):
    #     create_colorized_masks(val_mask_dir, os.path.join(base_dir, 'masks_val_color'))
        
    # if os.path.exists(test_mask_dir):
    #     create_colorized_masks(test_mask_dir, os.path.join(base_dir, 'masks_test_color'))
    
    print("\n[✓] All conversions completed successfully!")
    
    # Print class mapping for reference
    print("\nClass Mapping Reference:")
    print("=" * 50)
    for class_name, class_id in sorted(DACL10K_LABEL_MAP.items(), key=lambda x: x[1]):
        print(f"{class_id:2d}: {class_name}")
