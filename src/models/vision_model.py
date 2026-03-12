"""
Vision Model: CNN + BiLSTM + Attention
For video-based fall detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Attention mechanism for temporal features"""
    def __init__(self, hidden_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_output = torch.sum(lstm_output * attention_weights, dim=1)
        return weighted_output, attention_weights

class CNN2DFeatureExtractor(nn.Module):
    """2D CNN for extracting features from individual frames"""
    def __init__(self, output_dim=256):
        super(CNN2DFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return x

class VisionFallDetectionModel(nn.Module):
    """Complete Vision Model: CNN + BiLSTM + Attention"""
    def __init__(self, cnn_channels=[64, 128, 256], lstm_hidden=256, 
                 attention_dim=128, dropout=0.5):
        super(VisionFallDetectionModel, self).__init__()
        
        self.cnn = CNN2DFeatureExtractor(output_dim=256)
        self.cnn_output_size = 256
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.attention = AttentionLayer(lstm_hidden * 2, attention_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        batch_size, channels, time_steps, h, w = x.size()
        
        cnn_features = []
        for t in range(time_steps):
            frame = x[:, :, t, :, :]
            feat = self.cnn(frame)
            cnn_features.append(feat)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        attended_features, attention_weights = self.attention(lstm_out)
        output = self.classifier(attended_features)
        
        return output, attention_weights