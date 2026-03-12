import os
from pathlib import Path
import pandas as pd

def explore_sisfall_dataset():
    """Explore what's in the SisFall dataset"""
    
    print("\n" + "="*70)
    print("🔍 EXPLORING SISFALL DATASET")
    print("="*70)
    
    # Path to dataset
    project_root = Path(__file__).parent.parent
    sisfall_path = project_root / "datasets" / "raw" / "sisfall"
    
    # Check if dataset exists
    if not sisfall_path.exists():
        print("❌ SisFall dataset not found!")
        print(f"   Expected location: {sisfall_path}")
        print("\n   Please download and extract SisFall dataset first.")
        return
    
    print(f"\n📁 Dataset location: {sisfall_path}")
    
    # Count folders (subjects)
    subject_folders = [f for f in sisfall_path.iterdir() if f.is_dir()]
    print(f"\n👥 Number of subjects: {len(subject_folders)}")
    
    # Count files
    fall_files = []
    adl_files = []
    
    print("\n🔢 Counting files...")
    for subject_folder in subject_folders:
        for file in subject_folder.glob("*.txt"):
            filename = file.name
            if filename.startswith('F'):
                fall_files.append(file)
            elif filename.startswith('D'):
                adl_files.append(file)
    
    total_files = len(fall_files) + len(adl_files)
    
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    print(f"✓ Total files: {total_files}")
    print(f"✓ Fall samples: {len(fall_files)}")
    print(f"✓ Daily activity samples: {len(adl_files)}")
    print(f"✓ Balance: {len(fall_files)/total_files*100:.1f}% falls, {len(adl_files)/total_files*100:.1f}% ADLs")
    
    # Examine one file
    if fall_files:
        print("\n" + "="*70)
        print("📄 SAMPLE FILE EXAMINATION")
        print("="*70)
        
        sample_file = fall_files[0]
        print(f"\n📌 Looking at: {sample_file.name}")
        
        # Read first few lines
        try:
            with open(sample_file, 'r') as f:
                lines = f.readlines()[:10]
            
            print(f"✓ File has {len(lines)} lines (showing first 10)")
            print("\nSample data:")
            print("-" * 50)
            for i, line in enumerate(lines[:5], 1):
                print(f"Line {i}: {line.strip()}")
            print("-" * 50)
            
            # Try to parse as numbers
            first_line = lines[0].strip().split(',')
            if len(first_line) >= 6:
                print(f"\n📊 Data columns detected: {len(first_line)}")
                print("   Expected format:")
                print("   [AccX, AccY, AccZ, GyroX, GyroY, GyroZ]")
            
        except Exception as e:
            print(f"⚠️ Error reading file: {e}")
    
    print("\n" + "="*70)
    print("✅ DATASET EXPLORATION COMPLETE")
    print("="*70)
    print("\n💡 Next step: Run preprocessing to prepare data for training")
    print("   Command: python scripts/preprocess_sisfall.py")
    print("\n")
    
    return {
        'total_files': total_files,
        'fall_files': len(fall_files),
        'adl_files': len(adl_files),
        'subjects': len(subject_folders)
    }

if __name__ == "__main__":
    stats = explore_sisfall_dataset()