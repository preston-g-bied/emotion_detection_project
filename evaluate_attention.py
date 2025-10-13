"""
Evaluation script for Attention CNN model.
Compares performance with baseline CNN.
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json

# add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.preprocess import create_data_loaders
from src.models.attention_cnn import AttentionCNN

class AttentionEvaluator:
    """Evaluate attention model on test set."""

    def __init__(self, model_path, test_loader):
        self.test_loader = test_loader
        
        # setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')
        
        # load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = AttentionCNN(num_classes=7)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully (Validation Acc: {checkpoint['val_acc']:.2f}%)")
        
        # emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def predict(self):
        """Get predictions on test set."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\nRunning inference on test set...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)
                
                outputs = self.model(images, return_attention=False)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def compute_metrics(self, y_true, y_pred):
        """Compute classification metrics."""
        accuracy = 100. * np.sum(y_true == y_pred) / len(y_true)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        return accuracy, report
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='results/attention_confusion_matrix.png'):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # plot counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # plot normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels,
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")

    def plot_per_class_accuracy(self, report, save_path='results/attention_per_class_accuracy.png'):
        """Plot per-class performance metrics."""
        emotions = self.emotion_labels
        f1_scores = [report[emotion]['f1-score'] for emotion in emotions]
        precisions = [report[emotion]['precision'] for emotion in emotions]
        recalls = [report[emotion]['recall'] for emotion in emotions]
        
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics (Attention CNN)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class accuracy plot saved to {save_path}")

    def print_report(self, accuracy, report):
        """Print formatted evaluation report."""
        print("\n" + "="*70)
        print("ATTENTION CNN EVALUATION RESULTS")
        print("="*70)
        print(f"\nOverall Test Accuracy: {accuracy:.2f}%\n")
        print("-"*70)
        print(f"{'Emotion':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-"*70)
        
        for emotion in self.emotion_labels:
            metrics = report[emotion]
            print(f"{emotion:<12} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
                  f"{metrics['f1-score']:>10.3f} {int(metrics['support']):>10}")
        
        print("-"*70)
        macro_avg = report['macro avg']
        print(f"{'Macro Avg':<12} {macro_avg['precision']:>10.3f} {macro_avg['recall']:>10.3f} "
              f"{macro_avg['f1-score']:>10.3f}")
        print("="*70)
    
    def save_results(self, accuracy, report, save_path='results/attention_metrics.json'):
        """Save evaluation results to JSON."""
        results = {
            'overall_accuracy': float(accuracy),
            'per_class_metrics': {},
            'macro_avg': report['macro avg']
        }
        
        for emotion in self.emotion_labels:
            results['per_class_metrics'][emotion] = report[emotion]
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {save_path}")

    def compare_with_baseline(self, attention_acc, attention_report):
        """Compare attention model with baseline."""
        baseline_path = 'results/baseline_metrics.json'
        
        if not Path(baseline_path).exists():
            print("\nBaseline results not found. Skipping comparison.")
            return
        
        print("\n" + "="*70)
        print("COMPARISON WITH BASELINE")
        print("="*70)
        
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        baseline_acc = baseline['overall_accuracy']
        
        print(f"\nOverall Accuracy:")
        print(f"  Baseline:  {baseline_acc:.2f}%")
        print(f"  Attention: {attention_acc:.2f}%")
        print(f"  Improvement: {attention_acc - baseline_acc:+.2f}%")
        
        print(f"\nPer-Class F1-Score Comparison:")
        print("-"*70)
        print(f"{'Emotion':<12} {'Baseline':>12} {'Attention':>12} {'Improvement':>12}")
        print("-"*70)
        
        for emotion in self.emotion_labels:
            baseline_f1 = baseline['per_class_metrics'][emotion]['f1-score']
            attention_f1 = attention_report[emotion]['f1-score']
            improvement = attention_f1 - baseline_f1
            
            print(f"{emotion:<12} {baseline_f1:>12.3f} {attention_f1:>12.3f} {improvement:>+12.3f}")
        
        print("-"*70)
        baseline_macro = baseline['macro_avg']['f1-score']
        attention_macro = attention_report['macro avg']['f1-score']
        print(f"{'Macro Avg':<12} {baseline_macro:>12.3f} {attention_macro:>12.3f} {attention_macro - baseline_macro:>+12.3f}")
        print("="*70)

    def evaluate(self):
        """Run complete evaluation pipeline."""
        # get predictions
        y_true, y_pred, y_probs = self.predict()
        
        # compute metrics
        accuracy, report = self.compute_metrics(y_true, y_pred)
        
        # print report
        self.print_report(accuracy, report)
        
        # generate visualizations
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_per_class_accuracy(report)
        
        # save results
        self.save_results(accuracy, report)
        
        # compare with baseline
        self.compare_with_baseline(accuracy, report)
        
        return accuracy, report
    
def main():
    """Main evaluation function."""
    
    model_path = 'models/checkpoints/best_attention_model.pth'
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the attention model first using train_attention.py")
        return
    
    # load test data
    print("Loading test data...")
    loaders = create_data_loaders('data/processed', batch_size=64, num_workers=4)
    test_loader = loaders['test']
    
    print(f"Test set size: {len(test_loader.dataset)} images")
    
    # evaluate
    evaluator = AttentionEvaluator(model_path, test_loader)
    evaluator.evaluate()


if __name__ == "__main__":
    main()