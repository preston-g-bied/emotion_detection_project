"""
Real-time Emotion Detection with Webcam
Step 1: Basic webcam capture + emotion detection
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time
from torchvision import transforms

class RealtimeEmotionDetector:
    """
    Real-time emotion detection from webcam feed.
    Captures frames, preprocesses them, and runs emotion detection.
    """

    def __init__(self, model, device='cpu', target_fps=30):
        """
        Args:
            model: Trained emotion detection model
            device: Device to run inference on
            target_fps: Target frames per second
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps

        # emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 
                               'Neutral', 'Sad', 'Surprise']
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 255, 255),   # Yellow
            'Fear': (128, 0, 128),      # Purple
            'Happy': (0, 255, 0),       # Green
            'Neutral': (200, 200, 200), # Gray
            'Sad': (255, 0, 0),         # Blue
            'Surprise': (0, 165, 255)   # Orange
        }

        # face cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # performance tracking
        self.fps_history = []
        self.fps_window = 30    # average over last 30 frames

    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input.
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            Preprocessed tensor [1, 1, 48, 48]
        """
        # convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # apply transformations
        face_tensor = self.transform(face_rgb)
        face_tensor = face_tensor.unsqueeze(0)

        return face_tensor
    
    def detect_emotion(self, face_img):
        """
        Detect emotion from face image.
        
        Args:
            face_img: Face image
            
        Returns:
            emotion: Predicted emotion label
            confidence: Confidence score
            all_probs: All emotion probabilities
        """
        # preprocess
        face_tensor = self.preprocess_face(face_img).to(self.device)

        # inference
        with torch.no_grad():
            outputs = self.model(face_tensor, return_attention=False)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        emotion = self.emotion_labels[predicted.item()]
        confidence_score = confidence.item()
        all_probs = probs[0].cpu().numpy()

        return emotion, confidence_score, all_probs
    
    def draw_results(self, frame, faces, emotions_data):
        """
        Draw bounding boxes and emotion predictions on frame.
        
        Args:
            frame: Video frame
            faces: List of face bounding boxes
            emotions_data: List of (emotion, confidence, all_probs) tuples
            
        Returns:
            Annotated frame
        """
        for (x, y, w, h), (emotion, confidence, all_probs) in zip(faces, emotions_data):
            # draw bounding box
            color = self.emotion_colors[emotion]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # draw emotion label
            label = f'{emotion}: {confidence:.2%}'

            # background for text
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (x, y - text_h - 10),
                (x + text_w, y),
                color,
                -1
            )

            # draw rext
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # draw top 3 emotions with confidence bars
            top_3_idx = np.argsort(all_probs)[-3:][::-1]
            bar_x = x + w + 10
            bar_y = y
            bar_width = 150
            bar_height = 20

            for idx, emo_idx in enumerate(top_3_idx):
                emo_name = self.emotion_labels[emo_idx]
                prob = all_probs[emo_idx]

                # draw bar background
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y + idx * (bar_height + 5)),
                    (bar_x + bar_width, bar_y + idx * (bar_height + 5) + bar_height),
                    (50, 50, 50),
                    -1
                )

                # draw confidence bar
                bar_fill = int(bar_width * prob)
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y + idx * (bar_height + 5)),
                    (bar_x + bar_fill, bar_y + idx * (bar_height + 5) + bar_height),
                    self.emotion_colors[emo_name],
                    -1
                )

                # draw emotion name and percentage
                text = f'{emo_name}: {prob:.1%}'
                cv2.putText(
                    frame,
                    text,
                    (bar_x + 5, bar_y + idx + (bar_height + 5) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

        return frame
    
    def draw_fps(self, frame):
        """Draw FPS counter on frame."""
        if len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history[-self.fps_window:])
            fps_text = f'FPS: {avg_fps:.1f}'
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    def draw_instructions(self, frame):
        """Draw instructions on frame."""
        instructions = [
            "Presss 'q' to quit",
            "Press 's' to save screenshot"
        ]

        y_offset = frame.shape[0] - 60
        for instruction in instructions:
            cv2.putText(
                frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 25

    def run(self, camera_id=0, display_size=(1280, 720)):
        """
        Run real-time emotion detection.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            display_size: Size of display window
        """
        print("=" * 70)
        print("REAL-TIME EMOTION DETECTION")
        print("=" * 70)
        print(f"\nStarting webcam (device {camera_id})...")

        # open webcam
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"Webcam opened successfully")
        print(f"Target FPS: {self.target_fps}")
        print(f"Device: {self.device}")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("\nStarting detection...")

        screenshot_count = 0

        try:
            while True:
                frame_start = time.time()

                # capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    return
                
                # detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # process each face
                emotions_data = []
                for (x, y, w, h) in faces:
                    # extract face
                    face_img = frame[y:y+h, x:x+w]

                    # detect emotion
                    emotion, confidence, all_probs = self.detect_emotion(face_img)
                    emotions_data.append((emotion, confidence, all_probs))

                # draw results
                if len(faces) > 0:
                    frame = self.draw_results(frame, faces, emotions_data)
                else:
                    # no face detected
                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                # draw FPS and instructions
                self.draw_fps()
                self.draw_instructions()

                # resize of display
                frame_resized = cv2.resize(frame, display_size)

                # display
                cv2.imshow('Real-time Emotion Detection', frame_resized)

                # calculate FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(fps)

                # handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    screenshot_path = f'results/screenshot_{screenshot_count:03d}.png'
                    Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(screenshot_path, frame)
                    print(f'Screenshot saved: {screenshot_path}')
                    screenshot_count += 1

                # frame rate control
                elapsed = time.time() - frame_start
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # cleanup
            cap.release()
            cv2.destroyAllWindows()

            # Print statistics
            if len(self.fps_history) > 0:
                print("\n" + "=" * 70)
                print("SESSION STATISTICS")
                print("=" * 70)
                print(f"Average FPS: {np.mean(self.fps_history):.2f}")
                print(f"Min FPS: {np.min(self.fps_history):.2f}")
                print(f"Max FPS: {np.max(self.fps_history):.2f}")
                print(f"Total frames: {len(self.fps_history)}")
                print(f"Screenshots saved: {screenshot_count}")