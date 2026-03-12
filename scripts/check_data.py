from pathlib import Path
import os

def check_data():
    """Check if all processed data exists"""
    
    print("\n" + "="*70)
    print("🔍 CHECKING PROCESSED DATA FILES")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    combined_path = project_root / "datasets" / "combined"
    
    # Files to check
    required_files = {
        'SisFall (Sensor)': {
            'train': combined_path / "train" / "sisfall_data.pkl",
            'val': combined_path / "val" / "sisfall_data.pkl",
            'test': combined_path / "test" / "sisfall_data.pkl"
        },
        'UR Fall (Video)': {
            'train': combined_path / "train" / "urfall_video_data.pkl",
            'val': combined_path / "val" / "urfall_video_data.pkl",
            'test': combined_path / "test" / "urfall_video_data.pkl"
        }
    }
    
    print(f"\n📂 Checking: {combined_path}\n")
    
    all_good = True
    
    for dataset_name, files in required_files.items():
        print(f"{'='*70}")
        print(f"📦 {dataset_name}")
        print(f"{'='*70}")
        
        for split, filepath in files.items():
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✅ {split:6s}: {filepath.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ {split:6s}: MISSING - {filepath}")
                all_good = False
        print()
    
    if all_good:
        print("="*70)
        print("✅ ALL DATA FILES FOUND!")
        print("="*70)
        print("\n💡 You can now train models:")
        print("   • Sensor Model: python scripts/train_sensor_model.py")
        print("   • Vision Model: python scripts/train_vision_model.py")
    else:
        print("="*70)
        print("⚠️ SOME FILES ARE MISSING")
        print("="*70)
        print("\n💡 Run preprocessing for missing datasets:")
        print("   • SisFall: python scripts/preprocess_sisfall.py")
        print("   • UR Fall: python scripts/preprocess_urfall.py")
    
    print("\n")

if __name__ == "__main__":
    check_data()