"""
Fusion Model: Combines Vision and Sensor Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from vision_model import VisionFallDetectionModel
from sensor_model import SensorFallDetectionModel

class WeightedAverageFusion(nn.Module):
    """Weighted average fusion strategy"""
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

class MultiModalFallDetectionSystem(nn.Module):
    """Complete multi-modal fall detection system"""
    def __init__(self, vision_config, sensor_config, fusion_method='weighted_average'):
        super(MultiModalFallDetectionSystem, self).__init__()
        
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
        
        self.fusion_method = fusion_method
        self.fusion = WeightedAverageFusion()
    
    def forward(self, video_input, sensor_input):
        vision_output, vision_attention = self.vision_model(video_input)
        sensor_output, sensor_attention = self.sensor_model(sensor_input)
        
        vision_probs = F.softmax(vision_output, dim=1)
        sensor_probs = F.softmax(sensor_output, dim=1)
        
        fused_output = self.fusion(vision_probs, sensor_probs)
        
        return {
            'fused_output': fused_output,
            'vision_output': vision_output,
            'sensor_output': sensor_output,
            'vision_probs': vision_probs,
            'sensor_probs': sensor_probs,
            'vision_attention': vision_attention,
            'sensor_attention': sensor_attention
        }

def create_fusion_model(config):
    """Create fusion model from config"""
    model = MultiModalFallDetectionSystem(
        vision_config=config['model']['vision'],
        sensor_config=config['model']['sensor'],
        fusion_method=config['model']['fusion']['method']
    )
    return model