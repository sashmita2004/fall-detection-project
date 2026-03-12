"""
Video Data Preprocessing
Handles video frame extraction, resizing, and sequence creation
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import yaml

class VideoPreprocessor:
    def __init__(self, config_path='config.yaml'):
        """Initialize video preprocessor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.frame_width = self.config['data']['video']['frame_width']
        self.frame_height = self.config['data']['video']['frame_height']
        self.sequence_length = self.config['data']['video']['sequence_length']
        self.fps = self.config['data']['video']['fps']
        
    def extract_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def create_sequences(self, frames):
        """Create fixed-length sequences from frames"""
        sequences = []
        
        if len(frames) < self.sequence_length:
            # Pad with last frame if too short
            padding = np.repeat(frames[-1:], self.sequence_length - len(frames), axis=0)
            frames = np.concatenate([frames, padding])
        
        # Create overlapping sequences
        for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length // 2):
            seq = frames[i:i + self.sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def normalize_frames(self, frames):
        """Normalize pixel values to [0, 1]"""
        return frames.astype(np.float32) / 255.0
    
    def process_video(self, video_path, label):
        """Process single video file"""
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            return None
        
        # Create sequences
        sequences = self.create_sequences(frames)
        
        # Normalize
        sequences = self.normalize_frames(sequences)
        
        # Transpose to (N, C, T, H, W) format for PyTorch
        sequences = np.transpose(sequences, (0, 4, 1, 2, 3))
        
        return {
            'sequences': sequences,
            'label': label,
            'num_sequences': len(sequences)
        }
    
    def process_dataset(self, dataset_path, output_path):
        """Process entire dataset"""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        all_labels = []
        
        # Process falls
        print("Processing fall videos...")
        fall_videos = list((dataset_path / 'falls').glob('*.mp4'))
        for video_path in tqdm(fall_videos):
            result = self.process_video(video_path, label=1)  # 1 = fall
            if result:
                all_data.append(result['sequences'])
                all_labels.extend([1] * result['num_sequences'])
        
        # Process ADL (Activities of Daily Living)
        print("Processing ADL videos...")
        adl_videos = list((dataset_path / 'adl').glob('*.mp4'))
        for video_path in tqdm(adl_videos):
            result = self.process_video(video_path, label=0)  # 0 = not fall
            if result:
                all_data.append(result['sequences'])
                all_labels.extend([0] * result['num_sequences'])
        
        # Concatenate all data
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.array(all_labels)
        
        # Save processed data
        np.save(output_path / 'video_data.npy', all_data)
        np.save(output_path / 'video_labels.npy', all_labels)
        
        # Save metadata
        metadata = {
            'num_samples': len(all_labels),
            'num_falls': int(np.sum(all_labels)),
            'num_adl': int(len(all_labels) - np.sum(all_labels)),
            'sequence_shape': list(all_data.shape),
            'label_distribution': {
                'fall': int(np.sum(all_labels)),
                'adl': int(len(all_labels) - np.sum(all_labels))
            }
        }
        
        with open(output_path / 'video_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✅ Processed {len(all_labels)} sequences")
        print(f"   Falls: {metadata['num_falls']}")
        print(f"   ADL: {metadata['num_adl']}")
        print(f"   Saved to: {output_path}")
        
        return metadata

def main():
    """Main preprocessing function"""
    print("=" * 60)
    print("📹 VIDEO DATA PREPROCESSING")
    print("=" * 60)
    
    preprocessor = VideoPreprocessor()
    
    # Process UR Fall dataset
    print("\nProcessing UR Fall Dataset...")
    preprocessor.process_dataset(
        dataset_path='data/raw/urfall',
        output_path='data/processed/urfall'
    )
    
    print("\n" + "=" * 60)
    print("✅ VIDEO PREPROCESSING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()