# Emotion Detection with Spatial Attention

A PyTorch implementation of facial emotion recognition using CNN with spatial attention mechanisms. This project demonstrates how attention mechanisms can improve emotion classification by helping the model focus on important facial features.

## 🎯 Project Overview

This project implements and compares two CNN architectures for emotion recognition on the FER2013 dataset:
- **Baseline CNN**: Standard convolutional architecture (4 conv layers + 2 FC layers)
- **Attention CNN**: Same architecture enhanced with spatial attention mechanism

**Dataset**: FER2013 (35,887 facial images, 7 emotion classes)
- Training: 24,406 images
- Validation: 4,303 images  
- Test: 7,178 images

## 📊 Current Results

### Model Performance Comparison

| Model | Test Accuracy | Macro F1-Score | Parameters |
|-------|---------------|----------------|------------|
| Baseline CNN | 65.69% | 0.621 | 3,914,xxx |
| Attention CNN | **65.94%** | **0.629** | 3,948,011 |
| **Improvement** | **+0.25%** | **+0.008** | +33K (0.9%) |

### Per-Class Performance

| Emotion | Baseline F1 | Attention F1 | Change |
|---------|-------------|--------------|--------|
| Angry | 0.593 | 0.595 | +0.002 |
| Disgust | 0.536 | **0.565** | **+0.029** ✓ |
| Fear | 0.448 | **0.476** | **+0.028** ✓ |
| Happy | 0.852 | **0.872** | **+0.020** ✓ |
| Neutral | 0.628 | 0.629 | +0.001 |
| Sad | 0.534 | 0.491 | -0.043 |
| Surprise | 0.758 | 0.776 | +0.018 ✓ |

**Key Findings:**
- ✅ Significant improvements on difficult classes (Fear, Disgust)
- ✅ Happy emotion performance improved from 85.2% → 87.2%
- ✅ Minimal parameter overhead (only 0.9% increase)
- ⚠️ Trade-off: Sad emotion performance decreased (further investigation needed)

## 🏗️ Project Structure

```
emotion_detection_project/
├── src/
│   ├── data/
│   │   ├── preprocess.py          # Data loading and augmentation
│   │   └── data_utils.py          # Dataset utilities
│   └── models/
│       ├── baseline_cnn.py        # Baseline CNN architecture
│       ├── spatial_attention.py   # Attention module
│       └── attention_cnn.py       # Attention-enhanced CNN
├── data/
│   ├── raw/                       # Original FER2013 data
│   └── processed/                 # Preprocessed train/val/test splits
├── models/
│   └── checkpoints/               # Saved model checkpoints
│       ├── best_model.pth         # Baseline model
│       └── best_attention_model.pth
├── results/                       # Evaluation metrics and plots
│   ├── confusion_matrix.png
│   ├── attention_confusion_matrix.png
│   ├── baseline_metrics.json
│   └── attention_metrics.json
├── logs/                          # TensorBoard logs
├── train_baseline.py              # Baseline training script
├── train_attention.py             # Attention model training
├── evaluate_baseline.py           # Baseline evaluation
├── evaluate_attention.py          # Attention evaluation
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
CUDA (optional, for GPU training)
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion_detection_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the FER2013 dataset and place it in `data/raw/`

### Data Preprocessing

```bash
python run_preprocessing.py
```

This will:
- Organize data into train/val/test splits (85/15 split for train/val)
- Apply data augmentation
- Generate dataset statistics

## 🔧 Usage

### Training Models

**Train Baseline CNN:**
```bash
python train_baseline.py
```

**Train Attention CNN:**
```bash
python train_attention.py
```

Both scripts will:
- Train for 50 epochs
- Use Adam optimizer with learning rate scheduling
- Save checkpoints to `models/checkpoints/`
- Log training progress to TensorBoard

### Evaluating Models

**Evaluate Baseline:**
```bash
python evaluate_baseline.py
```

**Evaluate Attention (with comparison):**
```bash
python evaluate_attention.py
```

### Monitor Training

```bash
tensorboard --logdir=logs/
```

## 🧠 Model Architecture

### Baseline CNN
```
Input (1, 48, 48)
  ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
Conv2D(128) → BatchNorm → ReLU → MaxPool
Conv2D(256) → BatchNorm → ReLU → MaxPool
Conv2D(512) → BatchNorm → ReLU → MaxPool
  ↓
Flatten → FC(512) → ReLU → Dropout(0.5)
  ↓
FC(7) → Output
```

### Attention CNN
```
Input (1, 48, 48)
  ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
Conv2D(128) → BatchNorm → ReLU → MaxPool
Conv2D(256) → BatchNorm → ReLU → MaxPool
Conv2D(512) → BatchNorm → ReLU → MaxPool
  ↓
✨ Spatial Attention Module ✨
  ├─ Channel Attention (avg/max pooling + MLP)
  └─ Spatial Attention (conv on channel statistics)
  ↓
Flatten → FC(512) → ReLU → Dropout(0.5)
  ↓
FC(7) → Output
```

## 📈 Training Details

**Hyperparameters:**
- Optimizer: Adam
- Learning Rate: 0.001 (with ReduceLROnPlateau)
- Batch Size: 64
- Epochs: 50
- Weight Decay: 1e-4
- Dropout: 0.5

**Data Augmentation:**
- Random rotation (±10°)
- Random horizontal flip
- Normalization (mean=0.5, std=0.5)

**Hardware:**
- Device: MPS (Apple Silicon) / CUDA / CPU
- Training time: ~25 minutes per epoch on M1/M2

## 📝 Key Insights

### What Worked Well:
1. **Spatial attention improved difficult classes**: Fear and Disgust both saw ~3% F1-score improvements
2. **Minimal overhead**: Only 33K additional parameters (0.9% increase)
3. **Stable training**: Both models converged smoothly without instability
4. **Happy emotion boost**: Attention helped the model be more confident on positive emotions

### Areas for Improvement:
1. **Sad emotion**: Attention mechanism struggled, likely due to subtle facial features
2. **Class imbalance**: Disgust class has very few samples (111 in test set)
3. **Potential overfitting**: 5% gap between train and validation accuracy

## 🎯 Next Steps

- [x] Phase 1: Data preprocessing and exploration
- [x] Phase 2: Baseline CNN implementation
- [x] Phase 3: Spatial attention mechanism
- [ ] **Phase 4: Attention visualization** (In Progress)
  - Implement Grad-CAM visualization
  - Generate attention heatmaps
  - Overlay attention on facial images
  - Analyze what features the model focuses on
- [ ] Phase 5: Real-time webcam integration
- [ ] Phase 6: Web application deployment

## 📚 References

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## 📄 License

MIT License

## 👤 Author

Preston Bied

---

**Last Updated:** October 2025
**Status:** Phase 3 Complete - Moving to Attention Visualization