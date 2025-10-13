"""
Training script for Attention CNN model.
Modified from baseline training to work with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time
import json

# add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.preprocess import create_data_loaders
from src.models.attention_cnn import AttentionCNN

class AttentionTrainer:
    """Handles training and validation for attention model."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)
        print(f'Using device: {self.device}')
        
        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # setup directories
        self.checkpoint_dir = Path('models/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # tensorboard
        self.writer = SummaryWriter(f"logs/attention_{time.strftime('%Y%m%d_%H%M%S')}")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # forward pass (don't return attention during training)
            self.optimizer.zero_grad()
            outputs = self.model(images, return_attention=False)
            loss = self.criterion(outputs, labels)
            
            # backward pass
            loss.backward()
            self.optimizer.step()
            
            # statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images, return_attention=False)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        # save checkpoint
        checkpoint_path = self.checkpoint_dir / f'attention_checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_attention_model.pth'
            torch.save(checkpoint, best_path)
            print(f'Best model saved with validation accuracy: {val_acc:.2f}%')

    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Attention CNN Training")
        print("="*60)
        print(f"Total parameters: {self.model.get_num_params():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*60 + "\n")
        
        for epoch in range(self.config['num_epochs']):
            # train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # log to tensorboard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # print epoch summary
            print(f'\nEpoch {epoch+1}/{self.config["num_epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'  LR: {current_lr:.6f}')
            
            # save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*60)
        
        self.writer.close()
        
        # save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        
        Path('results').mkdir(exist_ok=True)
        with open('results/attention_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

def main():
    """Main training function."""

    # configuration
    config = {
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'batch_size': 64,
        'num_workers': 4
    }

    print('Configuration:')
    for key, value in config.items():
        print(f"  {key}: {value}")

    # load data
    print("\nLoading data...")
    loaders = create_data_loaders(
        'data/processed',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # create model
    print("\nInitializing Attention CNN model...")
    model = AttentionCNN(num_classes=7, dropout_rate=0.5)

    # create trainer and train
    trainer = AttentionTrainer(model, loaders['train'], loaders['val'], config)
    trainer.train()

if __name__ == "__main__":
    main()