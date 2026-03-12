import os
from pathlib import Path

# This is where your project is located
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# Information about each dataset
DATASET_INFO = {
    'urfall': {
        'name': 'UR Fall Detection Dataset',
        'size': '1.8 GB',
        'description': 'Video and sensor data of falls',
        'download_url': 'http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html'
    },
    'sisfall': {
        'name': 'SisFall Dataset', 
        'size': '350 MB',
        'description': 'Sensor data from accelerometer and gyroscope',
        'download_url': 'http://sistemic.udea.edu.co/en/research/projects/english-falls/'
    }
}

def create_folders():
    """Creates all the folders you need"""
    print("Creating folders for your project...")
    
    folders_to_create = [
        "datasets/raw/urfall",
        "datasets/raw/sisfall", 
        "datasets/processed/urfall",
        "datasets/processed/sisfall",
        "datasets/combined/train",
        "datasets/combined/val",
        "datasets/combined/test",
        "results/figures",
        "results/metrics",
        "results/logs"
    ]
    
    for folder in folders_to_create:
        folder_path = PROJECT_ROOT / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    print("\n✅ All folders created successfully!")

def show_dataset_info():
    """Shows information about datasets"""
    print("\n" + "="*60)
    print("DATASETS YOU NEED TO DOWNLOAD")
    print("="*60)
    
    for key, info in DATASET_INFO.items():
        print(f"\n📦 {info['name']}")
        print(f"   Size: {info['size']}")
        print(f"   Description: {info['description']}")
        print(f"   Download from: {info['download_url']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    create_folders()
    show_dataset_info()