"""
Video File Processor for Fall Detection
Processes video files and saves annotated output
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class VideoFallProcessor:
    """Process video files for fall detection"""
    
    def __init__(self, model_path, config_path='config.yaml'):
        """Initialize processor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.frame_width = self.config['data']['video']['frame_width']
        self.frame_height = self.config['data']['video']['frame_height']
        self.sequence_length = self.config['data']['video']['sequence_length']
        self.thresholds = self.config['alerts']['thresholds']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model...")
        from train_model import MultiModalModel
        
        self.model = MultiModalModel(
            self.config['model']['vision'],
            self.config['model']['sensor']
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded!")
        
        self.frame_buffer = []
    
    def preprocess_frame(self, frame):
        """Preprocess frame"""
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        frame = frame.astype(np.float32) / 255.0
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def simulate_sensor_data(self):
        """Simulate sensor data"""
        sensor_data = np.random.randn(100, 6).astype(np.float32)
        sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
        return sensor_data
    
    def get_alert_level(self, confidence):
        """Get alert level"""
        if confidence < self.thresholds['normal']:
            return 'NORMAL', (0, 255, 0)
        elif confidence < self.thresholds['warning']:
            return 'WARNING', (0, 165, 255)
        else:
            return 'CRITICAL', (0, 0, 255)
    
    def process_video(self, input_path, output_path):
        """Process video file"""
        print(f"\n📹 Processing video: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print("❌ Could not open video")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 150))
        
        frame_count = 0
        fall_detections = []
        
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess
            processed = self.preprocess_frame(frame)
            self.frame_buffer.append(processed)
            
            # Keep only last sequence_length frames
            if len(self.frame_buffer) > self.sequence_length:
                self.frame_buffer.pop(0)
            
            prediction = None
            confidence = 0.0
            
            if len(self.frame_buffer) == self.sequence_length:
                # Prepare input
                video_sequence = np.array(self.frame_buffer)
                video_sequence = np.transpose(video_sequence, (3, 0, 1, 2))
                video_tensor = torch.from_numpy(video_sequence).unsqueeze(0).to(self.device)
                
                sensor_data = self.simulate_sensor_data()
                sensor_tensor = torch.from_numpy(sensor_data).unsqueeze(0).to(self.device)
                
                # Detect
                with torch.no_grad():
                    fused_output, _, _ = self.model(video_tensor, sensor_tensor)
                    confidence = fused_output[0, 1].item()
                    prediction = 1 if confidence > 0.5 else 0
                
                if prediction == 1:
                    fall_detections.append({
                        'frame': frame_count,
                        'confidence': confidence,
                        'time': frame_count / fps
                    })
            
            # Draw info
            display_frame = self.draw_info(frame, prediction, confidence)
            
            # Write frame
            out.write(display_frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"\n✅ Processing complete!")
        print(f"   Output saved: {output_path}")
        print(f"   Falls detected: {len(fall_detections)}")
        
        if fall_detections:
            print("\n📊 Fall Detection Summary:")
            for i, detection in enumerate(fall_detections[:5], 1):
                print(f"   {i}. Frame {detection['frame']} ({detection['time']:.2f}s) - Confidence: {detection['confidence']*100:.1f}%")
    
    def draw_info(self, frame, prediction, confidence):
        """Draw info on frame"""
        h, w = frame.shape[:2]
        
        panel = np.zeros((150, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        if prediction is not None:
            alert_level, color = self.get_alert_level(confidence)
            
            status = "FALL DETECTED!" if prediction == 1 else "Normal Activity"
            cv2.putText(panel, status, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(panel, conf_text, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            alert_text = f"Alert: {alert_level}"
            cv2.putText(panel, alert_text, (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(panel, "Buffering...", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return np.vstack([frame, panel])
    

def main():
    """Main function"""
    print("=" * 60)
    print("🎬 VIDEO FALL DETECTION PROCESSOR")
    print("=" * 60)
    
    processor = VideoFallProcessor('models/fusion_model/best.pth')
    
    # Process sample video
    input_video = 'data/raw/urfall/falls/fall_1.mp4'
    output_video = 'outputs/demo_detection.mp4'
    
    Path('outputs').mkdir(exist_ok=True)
    
    processor.process_video(input_video, output_video)
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print(f"Watch the output: {output_video}")
    print("=" * 60)

if __name__ == "__main__":
    main()