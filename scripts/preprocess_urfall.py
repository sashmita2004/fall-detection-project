import os
import numpy as np
import cv2
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_video_frames(video_folder, max_frames=30, target_size=(224, 224)):
    """Extract frames from video folder"""
    frames = []
    
    # Get all image files
    image_files = sorted(list(video_folder.glob('*.png')) + 
                        list(video_folder.glob('*.jpg')) + 
                        list(video_folder.glob('*.bmp')))
    
    if len(image_files) == 0:
        return None
    
    # Sample frames evenly
    if len(image_files) > max_frames:
        indices = np.linspace(0, len(image_files)-1, max_frames, dtype=int)
        image_files = [image_files[i] for i in indices]
    
    for img_path in image_files:
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Resize
            img = cv2.resize(img, target_size)
            
            # Normalize
            img = img.astype('float32') / 255.0
            
            frames.append(img)
        except Exception as e:
            continue
    
    if len(frames) == 0:
        return None
    
    # Pad if necessary
    while len(frames) < max_frames:
        frames.append(np.zeros((target_size[0], target_size[1], 3)))
    
    return np.array(frames[:max_frames])

def preprocess_urfall():
    """Preprocess UR Fall Detection dataset"""
    
    print("\n" + "="*70)
    print("🔧 PREPROCESSING UR FALL DETECTION DATASET")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    raw_path = project_root / "datasets" / "raw" / "urfall"
    processed_path = project_root / "datasets" / "processed" / "urfall"
    combined_path = project_root / "datasets" / "combined"
    
    # Check if dataset exists
    if not raw_path.exists():
        print(f"\n❌ UR Fall dataset not found at: {raw_path}")
        print("\n💡 Please download and extract UR Fall dataset first")
        print("   Run: python scripts/download_urfall.py")
        return
    
    print(f"\n📂 Dataset location: {raw_path}")
    
    # Create processed directory
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Find all sequence folders
    fall_folders = sorted(raw_path.glob('fall-*'))
    adl_folders = sorted(raw_path.glob('adl-*'))
    
    print(f"\n📊 Found:")
    print(f"   Fall sequences: {len(fall_folders)}")
    print(f"   ADL sequences: {len(adl_folders)}")
    
    if len(fall_folders) == 0 and len(adl_folders) == 0:
        print("\n❌ No sequence folders found!")
        print("   Expected folders like: fall-01, fall-02, adl-01, adl-02")
        print("\n💡 Check your extraction - folders should be directly in urfall/")
        return
    
    # Process sequences
    video_data = []
    labels = []
    
    print("\n🎥 Processing fall sequences...")
    for folder in tqdm(fall_folders, desc="Falls"):
        rgb_folder = folder / 'rgb'
        if not rgb_folder.exists():
            rgb_folder = folder  # Sometimes images are directly in folder
        
        frames = extract_video_frames(rgb_folder)
        if frames is not None:
            video_data.append(frames)
            labels.append(1)  # Fall
    
    print("\n🎥 Processing ADL sequences...")
    for folder in tqdm(adl_folders, desc="ADLs"):
        rgb_folder = folder / 'rgb'
        if not rgb_folder.exists():
            rgb_folder = folder
        
        frames = extract_video_frames(rgb_folder)
        if frames is not None:
            video_data.append(frames)
            labels.append(0)  # ADL
    
    print(f"\n✓ Processed {len(video_data)} sequences")
    print(f"   Falls: {sum(labels)}")
    print(f"   ADLs: {len(labels) - sum(labels)}")
    
    if len(video_data) == 0:
        print("\n❌ No video data could be extracted!")
        return
    
    # Convert to arrays
    video_data = np.array(video_data)
    labels = np.array(labels)
    
    print(f"\n📐 Data shape: {video_data.shape}")
    
    # Split data
    print("\n📊 Splitting data...")
    
    # First split: train vs temp
    indices = np.arange(len(video_data))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Second split: val vs test
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"✓ Train: {len(train_idx)} sequences")
    print(f"✓ Val: {len(val_idx)} sequences")
    print(f"✓ Test: {len(test_idx)} sequences")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    
    data_dict = {
        'train_data': video_data[train_idx],
        'train_labels': labels[train_idx],
        'val_data': video_data[val_idx],
        'val_labels': labels[val_idx],
        'test_data': video_data[test_idx],
        'test_labels': labels[test_idx]
    }
    
    processed_file = processed_path / "urfall_processed.pkl"
    with open(processed_file, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"✓ Saved to: {processed_file}")
    
    # Save to combined folder
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_path = combined_path / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        
        split_data = {
            'data': video_data[split_idx],
            'labels': labels[split_idx],
            'source': 'urfall'
        }
        
        split_file = split_path / "urfall_data.pkl"
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        print(f"✓ Saved {split_name} split")
    
    print("\n" + "="*70)
    print("✅ UR FALL PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n📋 Summary:")
    print(f"   • Processed {len(video_data)} video sequences")
    print(f"   • Frame shape: {video_data.shape[1:]}")
    print(f"   • Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"   • Data saved to: {processed_path}")
    print("\n💡 Next: Train vision and gesture models")
    print("   Command: python scripts/train_vision_model.py")
    print("\n")

if __name__ == "__main__":
    preprocess_urfall()