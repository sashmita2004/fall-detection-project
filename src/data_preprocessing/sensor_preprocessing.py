"""
Sensor Data Preprocessing
Handles accelerometer and gyroscope data preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import yaml
from scipy import signal

class SensorPreprocessor:
    def __init__(self, config_path='config.yaml'):
        """Initialize sensor preprocessor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.window_size = self.config['data']['sensor']['window_size']
        self.sampling_rate = self.config['data']['sensor']['sampling_rate']
        self.features = self.config['data']['sensor']['features']
        
    def load_sensor_file(self, file_path):
        """Load sensor data from CSV file"""
        df = pd.read_csv(file_path)
        return df
    
    def apply_filters(self, data):
        """Apply filtering to sensor data"""
        # Low-pass filter to remove noise
        b, a = signal.butter(3, 0.1, btype='low')
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def create_windows(self, data, label):
        """Create sliding windows from sensor data"""
        windows = []
        labels = []
        
        step = self.window_size // 2  # 50% overlap
        
        for i in range(0, len(data) - self.window_size + 1, step):
            window = data[i:i + self.window_size]
            windows.append(window)
            labels.append(label)
        
        return np.array(windows), np.array(labels)
    
    def normalize_data(self, data):
        """Normalize sensor data"""
        # Z-score normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        return normalized
    
    def process_sensor_file(self, file_path, label):
        """Process single sensor file"""
        # Load data
        df = self.load_sensor_file(file_path)
        
        # Extract features
        data = df[self.features].values
        
        # Apply filters
        data = self.apply_filters(data)
        
        # Normalize
        data = self.normalize_data(data)
        
        # Create windows
        windows, labels = self.create_windows(data, label)
        
        return windows, labels
    
    def process_dataset(self, dataset_path, output_path):
        """Process entire sensor dataset"""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_windows = []
        all_labels = []
        
        # Process fall data
        print("Processing fall sensor data...")
        fall_files = list((dataset_path / 'falls').glob('*.csv'))
        for file_path in tqdm(fall_files):
            windows, labels = self.process_sensor_file(file_path, label=1)
            all_windows.append(windows)
            all_labels.append(labels)
        
        # Process ADL data
        print("Processing ADL sensor data...")
        adl_files = list((dataset_path / 'adl').glob('*.csv'))
        for file_path in tqdm(adl_files):
            windows, labels = self.process_sensor_file(file_path, label=0)
            all_windows.append(windows)
            all_labels.append(labels)
        
        # Concatenate all data
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Save processed data
        np.save(output_path / 'sensor_data.npy', all_windows)
        np.save(output_path / 'sensor_labels.npy', all_labels)
        
        # Save metadata
        metadata = {
            'num_samples': len(all_labels),
            'num_falls': int(np.sum(all_labels)),
            'num_adl': int(len(all_labels) - np.sum(all_labels)),
            'window_size': self.window_size,
            'num_features': len(self.features),
            'features': self.features,
            'data_shape': list(all_windows.shape)
        }
        
        with open(output_path / 'sensor_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✅ Processed {len(all_labels)} windows")
        print(f"   Falls: {metadata['num_falls']}")
        print(f"   ADL: {metadata['num_adl']}")
        print(f"   Saved to: {output_path}")
        
        return metadata

def main():
    """Main preprocessing function"""
    print("=" * 60)
    print("📊 SENSOR DATA PREPROCESSING")
    print("=" * 60)
    
    preprocessor = SensorPreprocessor()
    
    # Process SisFall dataset
    print("\nProcessing SisFall Dataset...")
    preprocessor.process_dataset(
        dataset_path='data/raw/sisfall',
        output_path='data/processed/sisfall'
    )
    
    print("\n" + "=" * 60)
    print("✅ SENSOR PREPROCESSING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()