"""
Real-Time Fall Detection System
Processes live video feed and detects falls
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import yaml
import time
from collections import deque
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class RealTimeFallDetector:
    """Real-time fall detection from video stream"""
    
    def __init__(self, model_path, config_path='config.yaml'):
        """Initialize the detector"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Video parameters
        self.frame_width = self.config['data']['video']['frame_width']
        self.frame_height = self.config['data']['video']['frame_height']
        self.sequence_length = self.config['data']['video']['sequence_length']
        
        # Alert thresholds
        self.thresholds = self.config['alerts']['thresholds']
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        from train_model import MultiModalModel
        
        self.model = MultiModalModel(
            self.config['model']['vision'],
            self.config['model']['sensor']
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Frame buffer for sequence
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Sensor buffer (simulated)
        self.sensor_buffer = deque(maxlen=100)
        
        # Alert state
        self.last_alert_time = 0
        self.cooldown = self.config['alerts']['cooldown_seconds']
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Resize
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def simulate_sensor_data(self):
        """Simulate sensor data (for demo without real sensors)"""
        # Generate random sensor data (6 channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        sensor_data = np.random.randn(100, 6).astype(np.float32)
        
        # Normalize
        sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
        
        return sensor_data
    
    def get_alert_level(self, confidence):
        """Determine alert level"""
        if confidence < self.thresholds['normal']:
            return 'NORMAL', (0, 255, 0)  # Green
        elif confidence < self.thresholds['warning']:
            return 'WARNING', (0, 165, 255)  # Orange
        else:
            return 'CRITICAL', (0, 0, 255)  # Red
    
    def detect_fall(self, frame):
        """Detect fall in current frame"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Need full sequence
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0, 'NORMAL', (0, 255, 0)
        
        # Prepare video input
        video_sequence = np.array(list(self.frame_buffer))
        video_sequence = np.transpose(video_sequence, (3, 0, 1, 2))  # (C, T, H, W)
        video_tensor = torch.from_numpy(video_sequence).unsqueeze(0).to(self.device)
        
        # Prepare sensor input (simulated)
        sensor_data = self.simulate_sensor_data()
        sensor_tensor = torch.from_numpy(sensor_data).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            fused_output, _, _ = self.model(video_tensor, sensor_tensor)
            
            fall_probability = fused_output[0, 1].item()
            prediction = 1 if fall_probability > 0.5 else 0
        
        # Get alert level
        alert_level, color = self.get_alert_level(fall_probability)
        
        return prediction, fall_probability, alert_level, color
    
    def draw_info(self, frame, prediction, confidence, alert_level, color, fps):
        """Draw information on frame"""
        h, w = frame.shape[:2]
        
        # Create info panel
        panel_height = 150
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # Status
        status_text = "FALL DETECTED!" if prediction == 1 else "Normal Activity"
        cv2.putText(panel, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_BOLD, 1.2, color, 3)
        
        # Confidence
        conf_text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(panel, conf_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Alert level
        alert_text = f"Alert Level: {alert_level}"
        cv2.putText(panel, alert_text, (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(panel, fps_text, (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine frame and panel
        combined = np.vstack([frame, panel])
        
        return combined
    
    def run(self, source=0):
        """Run real-time detection"""
        print("\n" + "=" * 60)
        print("🎥 STARTING REAL-TIME FALL DETECTION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Video source: {source}")
        print("Press 'q' to quit")
        print("=" * 60 + "\n")
        
        # Open video source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # FPS calculation
        fps_buffer = deque(maxlen=30)
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect fall
            prediction, confidence, alert_level, color = self.detect_fall(frame)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time + 1e-6)
            fps_buffer.append(fps)
            avg_fps = np.mean(fps_buffer)
            
            # Draw information
            if prediction is not None:
                display_frame = self.draw_info(
                    frame, prediction, confidence, 
                    alert_level, color, avg_fps
                )
            else:
                # Still buffering
                display_frame = frame.copy()
                cv2.putText(display_frame, "Initializing...", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Fall Detection System', display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Real-time detection stopped")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time fall detection')
    parser.add_argument('--model', type=str, default='models/fusion_model/best.pth',
                       help='Path to trained model')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or video file path)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Create detector
    detector = RealTimeFallDetector(args.model)
    
    # Run detection
    detector.run(source)

if __name__ == "__main__":
    main()