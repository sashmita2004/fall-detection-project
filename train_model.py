"""
Simplified Training Script for Multi-Modal Fall Detection
Self-contained script that avoids import issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import yaml
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import models directly with absolute imports
from src.models.vision_model import VisionFallDetectionModel
from src.models.sensor_model import SensorFallDetectionModel

class WeightedAverageFusion(nn.Module):
    """Weighted average fusion"""
    def __init__(self, vision_weight=0.6, sensor_weight=0.4):
        super(WeightedAverageFusion, self).__init__()
        self.vision_weight = nn.Parameter(torch.tensor(vision_weight))
        self.sensor_weight = nn.Parameter(torch.tensor(sensor_weight))
    
    def forward(self, vision_output, sensor_output):
        total = self.vision_weight + self.sensor_weight
        w_vision = self.vision_weight / total
        w_sensor = self.sensor_weight / total
        fused = w_vision * vision_output + w_sensor * sensor_output
        return fused

class MultiModalModel(nn.Module):
    """Multi-modal fusion model"""
    def __init__(self, vision_config, sensor_config):
        super(MultiModalModel, self).__init__()
        
        self.vision_model = VisionFallDetectionModel(
            cnn_channels=vision_config['cnn_channels'],
            lstm_hidden=vision_config['lstm_hidden'],
            attention_dim=vision_config['attention_dim'],
            dropout=vision_config['dropout']
        )
        
        self.sensor_model = SensorFallDetectionModel(
            input_dim=sensor_config['input_dim'],
            hidden_dim=sensor_config['hidden_dim'],
            num_layers=sensor_config['num_layers'],
            dropout=0.5
        )
        
        self.fusion = WeightedAverageFusion()
    
    def forward(self, video_input, sensor_input):
        vision_output, vision_attention = self.vision_model(video_input)
        sensor_output, sensor_attention = self.sensor_model(sensor_input)
        
        vision_probs = torch.softmax(vision_output, dim=1)
        sensor_probs = torch.softmax(sensor_output, dim=1)
        
        fused_output = self.fusion(vision_probs, sensor_probs)
        
        return fused_output, vision_output, sensor_output

class MultiModalDataset(Dataset):
    """Dataset loader"""
    def __init__(self, video_path, sensor_path):
        self.video_data = np.load(Path(video_path) / 'video_data.npy')
        self.video_labels = np.load(Path(video_path) / 'video_labels.npy')
        self.sensor_data = np.load(Path(sensor_path) / 'sensor_data.npy')
        self.sensor_labels = np.load(Path(sensor_path) / 'sensor_labels.npy')
        
        self.length = min(len(self.video_labels), len(self.sensor_labels))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        video = torch.from_numpy(self.video_data[idx]).float()
        sensor = torch.from_numpy(self.sensor_data[idx]).float()
        label = torch.tensor(self.video_labels[idx], dtype=torch.long)
        return video, sensor, label

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for video, sensor, labels in pbar:
        video = video.to(device)
        sensor = sensor.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        fused_output, vision_output, sensor_output = model(video, sensor)
        
        # Calculate losses
        loss_vision = criterion(vision_output, labels)
        loss_sensor = criterion(sensor_output, labels)
        fused_logits = torch.log(fused_output + 1e-8)
        loss_fused = criterion(fused_logits, labels)
        
        loss = 0.5 * loss_fused + 0.25 * loss_vision + 0.25 * loss_sensor
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(fused_output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for video, sensor, labels in pbar:
            video = video.to(device)
            sensor = sensor.to(device)
            labels = labels.to(device)
            
            fused_output, _, _ = model(video, sensor)
            
            fused_logits = torch.log(fused_output + 1e-8)
            loss = criterion(fused_logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(fused_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(val_loader), 100. * correct / total

def main():
    """Main training function"""
    print("=" * 60)
    print("🚀 FALL DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    
    # Load dataset
    print("\n📊 Loading datasets...")
    dataset = MultiModalDataset('data/processed/urfall', 'data/processed/sisfall')
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    print("\n🧠 Creating model...")
    model = MultiModalModel(config['model']['vision'], config['model']['sensor'])
    model = model.to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20  # Reduced for quick training
    best_val_acc = 0.0
    
    print(f"\n📚 Training for {num_epochs} epochs...")
    print("=" * 60 + "\n")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('models/fusion_model').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/fusion_model/best.pth')
            print(f"  💾 Saved best model!")
    
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()