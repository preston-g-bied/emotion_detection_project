import torch
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.attention_cnn import AttentionCNN
from src.realtime.realtime_emotion_detector import RealtimeEmotionDetector

def main():
    print("=" * 70)
    print("REAL-TIME EMOTION DETECTION - TEST")
    print("=" * 70)
    print("\nThis will open your webcam and detect emotions in real-time")
    print("Features:")
    print("  • Face detection using OpenCV")
    print("  • Real-time emotion prediction")
    print("  • Confidence scores and top-3 emotions")
    print("  • FPS counter")
    print("\n")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = 'models/checkpoints/best_attention_model.pth'
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the attention model first using train_attention.py")
        return
    
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model = AttentionCNN(num_classes=7)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully (Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    # Create real-time detector
    print("\nInitializing real-time detector...")
    detector = RealtimeEmotionDetector(
        model=model,
        device=device,
        target_fps=30
    )
    
    # Run detection
    print("\n" + "=" * 70)
    print("Starting webcam...")
    print("=" * 70)
    
    try:
        detector.run(camera_id=0, display_size=(1280, 720))
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure your webcam is connected")
        print("  • Close other applications using the webcam")
        print("  • Try a different camera_id (0, 1, 2, etc.)")


if __name__ == "__main__":
    main()