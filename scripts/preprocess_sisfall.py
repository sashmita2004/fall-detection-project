import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

def preprocess_sisfall():
    """Preprocess SisFall dataset for training"""
    
    print("\n" + "="*70)
    print("🔧 PREPROCESSING SISFALL DATASET")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    raw_path = project_root / "datasets" / "raw" / "sisfall"
    processed_path = project_root / "datasets" / "processed" / "sisfall"
    combined_path = project_root / "datasets" / "combined"
    
    # Create processed directory
    processed_path.mkdir(parents=True, exist_ok=True)
    
    print("\n📂 Loading data from:", raw_path)
    
    # Storage for processed data
    all_data = []
    all_labels = []
    file_info = []
    
    # Get all subject folders
    subject_folders = sorted([f for f in raw_path.iterdir() if f.is_dir()])
    
    print(f"📊 Processing {len(subject_folders)} subjects...")
    
    processed_files = 0
    skipped_files = 0
    
    # Process each subject
    for subject_idx, subject_folder in enumerate(subject_folders, 1):
        subject_name = subject_folder.name
        print(f"\n  [{subject_idx}/{len(subject_folders)}] Processing {subject_name}...", end=" ")
        
        subject_file_count = 0
        
        # Get all txt files in this subject folder
        for file_path in subject_folder.glob("*.txt"):
            try:
                # Read the file
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse data
                data_rows = []
                for line in lines:
                    # Remove semicolon and split by comma
                    line = line.strip().replace(';', '')
                    if line:
                        values = [float(x.strip()) for x in line.split(',')]
                        # Take only first 6 columns (AccX, AccY, AccZ, GyroX, GyroY, GyroZ)
                        if len(values) >= 6:
                            data_rows.append(values[:6])
                
                if len(data_rows) > 0:
                    # Convert to numpy array
                    data_array = np.array(data_rows)
                    
                    # Determine label from filename
                    filename = file_path.name
                    if filename.startswith('F'):
                        label = 1  # Fall
                    elif filename.startswith('D'):
                        label = 0  # Daily activity
                    else:
                        continue  # Skip unknown files
                    
                    # Store data
                    all_data.append(data_array)
                    all_labels.append(label)
                    file_info.append({
                        'filename': filename,
                        'subject': subject_name,
                        'label': 'fall' if label == 1 else 'adl',
                        'samples': len(data_array)
                    })
                    
                    processed_files += 1
                    subject_file_count += 1
                
            except Exception as e:
                skipped_files += 1
                continue
        
        print(f"✓ ({subject_file_count} files)")
    
    print("\n" + "="*70)
    print("📊 PREPROCESSING STATISTICS")
    print("="*70)
    print(f"✓ Processed files: {processed_files}")
    print(f"⚠ Skipped files: {skipped_files}")
    print(f"✓ Total samples: {len(all_data)}")
    
    # Count falls and ADLs
    fall_count = sum(1 for label in all_labels if label == 1)
    adl_count = sum(1 for label in all_labels if label == 0)
    print(f"✓ Falls: {fall_count} ({fall_count/len(all_labels)*100:.1f}%)")
    print(f"✓ ADLs: {adl_count} ({adl_count/len(all_labels)*100:.1f}%)")
    
    # Normalize data (optional but recommended)
    print("\n🔧 Normalizing data...")
    normalized_data = []
    for data in all_data:
        # Simple normalization: scale to -1 to 1 range
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 0:
            normalized = 2 * (data - data_min) / (data_max - data_min) - 1
        else:
            normalized = data
        normalized_data.append(normalized)
    
    print("✓ Normalization complete")
    
    # Split into train/val/test (70/15/15)
    print("\n📊 Splitting data...")
    
    # Convert labels to numpy array
    all_labels = np.array(all_labels)
    
    # First split: train vs temp (70% vs 30%)
    indices = np.arange(len(normalized_data))
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=0.3, 
        random_state=42, 
        stratify=all_labels
    )
    
    # Second split: val vs test (15% vs 15% of total = 50/50 of temp)
    temp_labels = all_labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_labels
    )
    
    print(f"✓ Train samples: {len(train_idx)} ({len(train_idx)/len(all_data)*100:.1f}%)")
    print(f"✓ Val samples: {len(val_idx)} ({len(val_idx)/len(all_data)*100:.1f}%)")
    print(f"✓ Test samples: {len(test_idx)} ({len(test_idx)/len(all_data)*100:.1f}%)")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    
    # Save to processed folder
    data_dict = {
        'train_data': [normalized_data[i] for i in train_idx],
        'train_labels': all_labels[train_idx],
        'val_data': [normalized_data[i] for i in val_idx],
        'val_labels': all_labels[val_idx],
        'test_data': [normalized_data[i] for i in test_idx],
        'test_labels': all_labels[test_idx],
        'file_info': file_info
    }
    
    processed_file = processed_path / "sisfall_processed.pkl"
    with open(processed_file, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"✓ Saved to: {processed_file}")
    
    # Also save to combined folder for easy access
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_path = combined_path / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        
        split_data = {
            'data': [normalized_data[i] for i in split_idx],
            'labels': all_labels[split_idx],
            'source': 'sisfall'
        }
        
        split_file = split_path / "sisfall_data.pkl"
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        print(f"✓ Saved {split_name} split to: {split_file}")
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\n📋 Summary:")
    print(f"   • Processed {processed_files} files from {len(subject_folders)} subjects")
    print(f"   • Train: {len(train_idx)} samples")
    print(f"   • Val: {len(val_idx)} samples")
    print(f"   • Test: {len(test_idx)} samples")
    print(f"   • Data saved to: {processed_path}")
    print("\n💡 Next step: Train your model with real data!")
    print("   Command: python scripts/train_with_real_data.py")
    print("\n")
    
    return data_dict

if __name__ == "__main__":
    preprocess_sisfall()