# Real-Time Emotion Detection with Attention Mechanisms

A deep learning system that classifies facial emotions in real-time using CNNs with spatial attention mechanisms. Built as a portfolio project to explore attention-based architectures and their interpretability in computer vision tasks.

## What This Project Does

Detects and classifies facial emotions across 7 categories (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) using the FER2013 dataset. The model uses spatial attention to focus on relevant facial features, and I've built visualization tools to see exactly where the network is "looking" when making predictions.

**Real-time performance:** ~15-30 FPS on webcam with face detection and emotion prediction overlay.

## Results

| Model | Accuracy | Parameters | Key Improvement |
|-------|----------|------------|-----------------|
| Baseline CNN | 65.7% | 3.6M | Solid foundation |
| Attention CNN | 65.9% | 3.6M (+33K) | Better on Fear/Disgust |

The attention mechanism added only 0.9% more parameters but improved performance on challenging classes like Fear (+5%) and Disgust (+3.6%). While the overall accuracy gain is modest, the attention maps provide valuable interpretability—you can actually see which facial features drive each emotion prediction.

### Per-Class Performance (Attention Model)

- **Happy:** 87.2% F1 (easiest to detect)
- **Surprise:** 77.6% F1
- **Neutral:** 62.9% F1
- **Angry:** 59.5% F1
- **Disgust:** 56.5% F1 (hardest, only 111 test samples)
- **Fear:** 47.6% F1 (challenging but improved with attention)
- **Sad:** 49.1% F1

## Project Structure

```
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # CNN architectures (baseline + attention)
│   ├── utils/             # Training utilities
│   ├── visualization/     # Attention map generation
│   └── realtime/          # Webcam integration
│
├── demos/                 # Demo scripts
│   ├── demo_attention_viz.py
│   ├── demo_realtime.py
│   ├── demo_class_specific.py
│   └── demo_multi_layer.py
│
├── notebooks/             # Exploratory data analysis
├── results/               # Training plots and metrics
├── models/                # Saved checkpoints
└── data/                  # FER2013 dataset
```

## Quick Start

### Setup

```bash
# Clone and navigate to project
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Download the FER2013 dataset and place it in `data/external/`. Then run:

```bash
python run_preprocessing.py
```

This creates train/val/test splits in `data/processed/`.

### Training

**Baseline model:**
```bash
python train_baseline.py
```

**Attention model:**
```bash
python train_attention.py
```

Models are saved to `models/checkpoints/` with training logs in `logs/`.

### Evaluation

```bash
python evaluate_attention.py
```

Generates confusion matrices and per-class metrics in `results/plots/`.

### Demos

**Real-time webcam detection:**
```bash
python demos/demo_realtime.py
```

**Attention visualization:**
```bash
python demos/demo_attention_viz.py
```

**Class-specific attention patterns:**
```bash
python demos/demo_class_specific.py
```

## Technical Details

### Architecture

**Baseline CNN:**
- 4 convolutional blocks with batch normalization
- MaxPooling and dropout for regularization
- 2 fully connected layers
- ~3.6M parameters

**Attention CNN:**
- Same backbone as baseline
- Spatial attention module after each conv block
- Learns to weight feature maps by importance
- +33K parameters (0.9% increase)

### Training Setup

- **Optimizer:** Adam (lr=0.001)
- **Batch size:** 64
- **Augmentation:** Random horizontal flip, rotation (±10°), brightness/contrast
- **Hardware:** M4 Max with MPS acceleration (2-4x speedup vs CPU)
- **Training time:** ~15-20 minutes per model

### Attention Mechanism

The spatial attention module computes attention weights for each spatial location in the feature maps. This helps the model focus on discriminative facial regions (eyes, mouth, eyebrows) while suppressing background noise.

```python
attention_weights = sigmoid(conv(avg_pool + max_pool))
output = input * attention_weights
```

## What I Learned

**Attention isn't always a silver bullet.** The performance gain was modest (~0.25%), but the interpretability benefit is huge. Seeing where the model looks helps debug misclassifications and builds trust in predictions.

**Class imbalance matters.** Disgust has only 111 test samples vs 1774 for Happy. The model struggles with rare classes, and attention helps slightly but doesn't solve the fundamental data limitation.

**Real-time ML is different.** Building a 30 FPS webcam system taught me about inference optimization, face detection robustness, and the importance of smooth UI feedback.

**MPS acceleration on Apple Silicon is solid.** Got 2-4x speedup on M4 Max compared to CPU training. Not as fast as CUDA GPUs but great for local development.

## Future Improvements

- [ ] Implement class weighting or focal loss for imbalanced classes
- [ ] Try channel attention (CBAM) in addition to spatial attention
- [ ] Multi-face detection and emotion tracking
- [ ] Web deployment with FastAPI + React frontend
- [ ] Cross-dataset validation (FER2013 vs AffectNet)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy, Matplotlib, Seaborn
- tqdm

See `requirements.txt` for full list.

## Acknowledgments

- **Dataset:** FER2013 from Kaggle
- **Attention mechanism:** Inspired by CBAM (Convolutional Block Attention Module)
- **Face detection:** OpenCV Haar Cascades

## License

MIT License - feel free to use this code for your own projects.