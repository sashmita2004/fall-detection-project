"""
Dataset Download and Organization Script
"""

import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

# Create data directories
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def setup_dataset_structure():
    """Create dataset folder structure"""
    datasets = {
        'urfall': DATA_DIR / 'urfall',
        'le2i': DATA_DIR / 'le2i',
        'sisfall': DATA_DIR / 'sisfall'
    }
    
    for name, path in datasets.items():
        path.mkdir(parents=True, exist_ok=True)
        (path / 'falls').mkdir(exist_ok=True)
        (path / 'adl').mkdir(exist_ok=True)
        print(f"✅ Created structure for: {name}")
    
    return datasets

def create_sample_video(path, frames=30, label='fall'):
    """Create a sample video file"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (640, 480))
    
    for i in range(frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        if label == 'fall':
            cv2.circle(frame, (320, 240 - i*5), 50, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (320 + i*10, 240), 50, (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()

def create_sample_sensor_data(path, samples=100, label='fall'):
    """Create sample sensor data"""
    time = np.arange(samples) * 0.02
    
    if label == 'fall':
        acc_x = np.random.normal(0, 0.5, samples)
        acc_y = np.random.normal(0, 0.5, samples)
        acc_z = np.random.normal(9.8, 0.5, samples)
        
        fall_point = samples // 2
        acc_x[fall_point-5:fall_point+5] += np.random.uniform(15, 25, 10)
        acc_z[fall_point-5:fall_point+5] -= np.random.uniform(5, 10, 10)
    else:
        acc_x = np.random.normal(0, 0.2, samples)
        acc_y = np.random.normal(0, 0.2, samples)
        acc_z = np.random.normal(9.8, 0.3, samples)
    
    gyro_x = np.random.normal(0, 0.1, samples)
    gyro_y = np.random.normal(0, 0.1, samples)
    gyro_z = np.random.normal(0, 0.1, samples)
    
    df = pd.DataFrame({
        'time': time,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'label': label
    })
    
    df.to_csv(path, index=False)

def create_sample_data(datasets):
    """Create sample synthetic data for testing"""
    print("Creating sample video data...")
    
    urfall_path = datasets['urfall']
    
    for i in range(5):
        create_sample_video(
            urfall_path / 'falls' / f'fall_{i+1}.mp4',
            frames=30,
            label='fall'
        )
    
    for i in range(5):
        create_sample_video(
            urfall_path / 'adl' / f'adl_{i+1}.mp4',
            frames=30,
            label='adl'
        )
    
    print("Creating sample sensor data...")
    
    sisfall_path = datasets['sisfall']
    
    for i in range(10):
        label = 'fall' if i < 5 else 'adl'
        folder = sisfall_path / ('falls' if label == 'fall' else 'adl')
        create_sample_sensor_data(
            folder / f'sensor_{i+1}.csv',
            samples=100,
            label=label
        )
    
    print("✅ Sample data created successfully!")

def main():
    """Main function"""
    print("=" * 60)
    print("🚑 FALL DETECTION DATASET SETUP")
    print("=" * 60)
    
    print("\n📁 Creating folder structure...")
    datasets = setup_dataset_structure()
    
    print("\n🔄 Creating sample data for testing...")
    create_sample_data(datasets)
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print(f"\nData location: {DATA_DIR.absolute()}")

if __name__ == "__main__":
    main()