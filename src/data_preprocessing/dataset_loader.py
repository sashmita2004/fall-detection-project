"""
Dataset Loaders for Fall Detection
PyTorch Dataset classes for video and sensor data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json

class VideoFallDataset(Dataset):
    """Dataset for video-based fall detection"""
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        
        # Load preprocessed data
        self.data = np.load(self.data_path / 'video_data.npy')
        self.labels = np.load(self.data_path / 'video_labels.npy')
        
        # Load metadata
        with open(self.data_path / 'video_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.transform = transform
        
        print(f"Loaded video dataset:")
        print(f"  Samples: {len(self.labels)}")
        print(f"  Falls: {self.metadata['num_falls']}")
        print(f"  ADL: {self.metadata['num_adl']}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get video sequence and label
        video = self.data[idx]
        label = self.labels[idx]
        
        # Convert to torch tensors
        video = torch.from_numpy(video).float()
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            video = self.transform(video)
        
        return video, label

class SensorFallDataset(Dataset):
    """Dataset for sensor-based fall detection"""
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        
        # Load preprocessed data
        self.data = np.load(self.data_path / 'sensor_data.npy')
        self.labels = np.load(self.data_path / 'sensor_labels.npy')
        
        # Load metadata
        with open(self.data_path / 'sensor_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.transform = transform
        
        print(f"Loaded sensor dataset:")
        print(f"  Samples: {len(self.labels)}")
        print(f"  Falls: {self.metadata['num_falls']}")
        print(f"  ADL: {self.metadata['num_adl']}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get sensor window and label
        sensor = self.data[idx]
        label = self.labels[idx]
        
        # Convert to torch tensors
        sensor = torch.from_numpy(sensor).float()
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            sensor = self.transform(sensor)
        
        return sensor, label

class MultiModalFallDataset(Dataset):
    """Combined dataset for multi-modal training"""
    def __init__(self, video_path, sensor_path):
        self.video_dataset = VideoFallDataset(video_path)
        self.sensor_dataset = SensorFallDataset(sensor_path)
        
        # Ensure same number of samples (take minimum)
        self.length = min(len(self.video_dataset), len(self.sensor_dataset))
        
        print(f"\nMulti-modal dataset created:")
        print(f"  Total samples: {self.length}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        video, video_label = self.video_dataset[idx]
        sensor, sensor_label = self.sensor_dataset[idx]
        
        # Labels should match (both fall or both non-fall)
        # Use video label as primary
        return video, sensor, video_label

def create_data_loaders(config, video_path, sensor_path, batch_size=16, 
                       train_split=0.7, val_split=0.15):
    """Create train, validation, and test data loaders"""
    
    # Load datasets
    video_dataset = VideoFallDataset(video_path)
    sensor_dataset = SensorFallDataset(sensor_path)
    multimodal_dataset = MultiModalFallDataset(video_path, sensor_path)
    
    # Calculate splits
    dataset_size = len(multimodal_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"\nDataset splits:")
    print(f"  Train: {train_size} ({train_split*100:.0f}%)")
    print(f"  Val:   {val_size} ({val_split*100:.0f}%)")
    print(f"  Test:  {test_size} ({(1-train_split-val_split)*100:.0f}%)")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        multimodal_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'sizes': {
            'train': train_size,
            'val': val_size,
            'test': test_size
        }
    }

# Test the dataset loaders
if __name__ == "__main__":
    print("=" * 60)
    print("📚 TESTING DATASET LOADERS")
    print("=" * 60)
    
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    print("\nCreating data loaders...")
    loaders = create_data_loaders(
        config,
        video_path='data/processed/urfall',
        sensor_path='data/processed/sisfall',
        batch_size=4
    )
    
    # Test train loader
    print("\n" + "-" * 60)
    print("Testing train loader...")
    print("-" * 60)
    
    train_loader = loaders['train']
    
    for batch_idx, (video, sensor, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Video shape: {video.shape}")
        print(f"  Sensor shape: {sensor.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Falls in batch: {labels.sum().item()}")
        
        if batch_idx >= 1:  # Only show 2 batches
            break
    
    print("\n" + "=" * 60)
    print("✅ DATASET LOADERS TEST PASSED!")
    print("=" * 60)
    
    print(f"\nDataset Summary:")
    print(f"  Total batches (train): {len(loaders['train'])}")
    print(f"  Total batches (val): {len(loaders['val'])}")
    print(f"  Total batches (test): {len(loaders['test'])}")