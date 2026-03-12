"""
Hand Gesture Detection for Emergency Signals
Detects hand gestures like raised hand, waving, help signals
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class HandGestureDetector:
    """Detect emergency hand gestures"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture history for stability
        self.gesture_history = deque(maxlen=10)
        
        # Emergency gestures
        self.emergency_gestures = {
            'HELP_SIGNAL': 0,  # Hand raised with open palm
            'WAVING': 0,       # Waving motion
            'BOTH_HANDS_UP': 0, # Both hands raised
            'STOP_GESTURE': 0   # Stop sign with palm
        }
        
    def detect_landmarks(self, frame):
        """Detect hand landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.hands.process(rgb_frame)
        
        return results
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_hand_open(self, landmarks):
        """Check if hand is open (all fingers extended)"""
        # Get fingertip and base landmarks
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_bases = [5, 9, 13, 17]  # Finger bases
        
        fingers_extended = 0
        
        # Check thumb separately
        thumb_tip = landmarks.landmark[4]
        thumb_base = landmarks.landmark[2]
        if thumb_tip.x > thumb_base.x:  # Thumb extended (assuming right hand)
            fingers_extended += 1
        
        # Check other fingers
        for tip, base in zip(finger_tips, finger_bases):
            if landmarks.landmark[tip].y < landmarks.landmark[base].y:
                fingers_extended += 1
        
        return fingers_extended >= 4  # At least 4 fingers extended
    
    def is_hand_raised(self, landmarks, frame_height):
        """Check if hand is raised above shoulder level"""
        wrist = landmarks.landmark[0]
        # Consider hand raised if wrist is in upper 40% of frame
        return wrist.y < 0.4
    
    def detect_waving(self, landmarks):
        """Detect waving motion by tracking wrist movement"""
        wrist = landmarks.landmark[0]
        
        # Add to history
        self.gesture_history.append(wrist.x)
        
        if len(self.gesture_history) >= 10:
            # Check for oscillating movement
            positions = list(self.gesture_history)
            changes = np.diff(positions)
            
            # Count direction changes
            direction_changes = np.sum(np.abs(np.diff(np.sign(changes))))
            
            # If wrist moves left-right-left or right-left-right
            return direction_changes >= 3
        
        return False
    
    def detect_stop_gesture(self, landmarks):
        """Detect stop gesture (open palm facing camera)"""
        # Check if hand is open and palm is facing camera
        # This is approximated by checking if fingers are extended and visible
        return self.is_hand_open(landmarks)
    
    def detect_gesture(self, frame):
        """Main gesture detection function"""
        results = self.detect_landmarks(frame)
        
        detected_gesture = None
        confidence = 0.0
        emergency = False
        hand_landmarks = None
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # Both hands raised - HIGH EMERGENCY
            if num_hands == 2:
                both_raised = True
                for hand_lms in results.multi_hand_landmarks:
                    if not self.is_hand_raised(hand_lms, frame.shape[0]):
                        both_raised = False
                        break
                
                if both_raised:
                    detected_gesture = "BOTH HANDS UP"
                    confidence = 0.95
                    emergency = True
                    hand_landmarks = results.multi_hand_landmarks
            
            # Single hand gestures
            if not detected_gesture:
                for hand_lms in results.multi_hand_landmarks:
                    hand_landmarks = [hand_lms]
                    
                    # Check for raised hand with open palm
                    if self.is_hand_raised(hand_lms, frame.shape[0]) and self.is_hand_open(hand_lms):
                        detected_gesture = "HELP SIGNAL"
                        confidence = 0.90
                        emergency = True
                        break
                    
                    # Check for waving
                    elif self.detect_waving(hand_lms):
                        detected_gesture = "WAVING"
                        confidence = 0.85
                        emergency = True
                        break
                    
                    # Check for stop gesture
                    elif self.detect_stop_gesture(hand_lms) and self.is_hand_raised(hand_lms, frame.shape[0]):
                        detected_gesture = "STOP SIGNAL"
                        confidence = 0.88
                        emergency = True
                        break
        
        return {
            'gesture': detected_gesture,
            'confidence': confidence,
            'emergency': emergency,
            'landmarks': hand_landmarks,
            'num_hands': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        }
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        if landmarks:
            for hand_lms in landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        return frame
    
    def draw_gesture_info(self, frame, gesture_result):
        """Draw gesture information on frame"""
        if gesture_result['gesture']:
            h, w = frame.shape[:2]
            
            # Draw emergency banner if detected
            if gesture_result['emergency']:
                # Red banner
                cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 200), -1)
                
                # Text
                text = f"🚨 {gesture_result['gesture']}"
                cv2.putText(frame, text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Draw confidence
            conf_text = f"Confidence: {gesture_result['confidence']*100:.0f}%"
            cv2.putText(frame, conf_text, (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

# Test the detector
if __name__ == "__main__":
    print("=" * 60)
    print("🖐️ TESTING HAND GESTURE DETECTOR")
    print("=" * 60)
    
    detector = HandGestureDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
    else:
        print("✅ Camera opened. Press 'q' to quit.")
        print("\nTry these gestures:")
        print("  - Raise hand with open palm (HELP)")
        print("  - Wave your hand (WAVING)")
        print("  - Raise both hands (BOTH HANDS UP)")
        print("  - Show stop gesture (STOP)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect gesture
            result = detector.detect_gesture(frame)
            
            # Draw landmarks
            if result['landmarks']:
                frame = detector.draw_landmarks(frame, result['landmarks'])
            
            # Draw info
            frame = detector.draw_gesture_info(frame, result)
            
            # Show
            cv2.imshow('Hand Gesture Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()