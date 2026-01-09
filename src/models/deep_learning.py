"""
Deep Learning Models for Driver Behavior Classification

Implements:
- Sequence Data Loader (sliding windows)
- LSTM (Long Short-Term Memory) model
- Transformer Encoder model

Usage:
    python -m src.models.deep_learning --train data/processed/training_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm


# Feature columns for the models
FEATURE_COLUMNS = [
    'speed', 'steering', 'accel_forward', 'accel_lateral', 
    'gyro_yaw', 'radar_distance', 'vehicle_count',
    'speed_change', 'steering_rate', 'steering_jerk'
]


class SequenceDataset(Dataset):
    """
    Dataset for sequence-based deep learning.
    
    Creates sliding windows of driving data for temporal analysis.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        sequence_length: int = 20,
        feature_cols: List[str] = None,
        scaler: StandardScaler = None,
        label_encoder: LabelEncoder = None,
        fit_transforms: bool = False
    ):
        """
        Initialize sequence dataset.
        
        Args:
            df: DataFrame with features and 'label' column
            sequence_length: Number of timesteps per sequence
            feature_cols: Feature columns to use
            scaler: Pre-fitted scaler (or None to fit new one)
            label_encoder: Pre-fitted label encoder (or None)
            fit_transforms: Whether to fit the transforms
        """
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        
        # Filter to available columns
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        
        # Extract features and labels
        X = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y = df['label'].values
        
        # Handle transforms
        self.scaler = scaler if scaler else StandardScaler()
        self.label_encoder = label_encoder if label_encoder else LabelEncoder()
        
        if fit_transforms:
            X = self.scaler.fit_transform(X)
            y = self.label_encoder.fit_transform(y)
        else:
            X = self.scaler.transform(X)
            y = self.label_encoder.transform(y)
        
        # Create sequences using sliding window
        self.sequences = []
        self.labels = []
        
        # Group by segment if available
        if 'segment_id' in df.columns:
            for seg_id in df['segment_id'].unique():
                seg_mask = df['segment_id'] == seg_id
                seg_X = X[seg_mask]
                seg_y = y[seg_mask]
                self._create_sequences(seg_X, seg_y)
        else:
            self._create_sequences(X, y)
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sliding window sequences."""
        for i in range(len(X) - self.sequence_length):
            self.sequences.append(X[i:i+self.sequence_length])
            # Use the label of the last timestep
            self.labels.append(y[i+self.sequence_length-1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


class LSTMClassifier(nn.Module):
    """
    LSTM-based driver behavior classifier.
    
    Uses bidirectional LSTM layers to capture temporal patterns
    in driving behavior.
    """
    
    def __init__(
        self, 
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state
        # For bidirectional, concatenate forward and backward
        if self.bidirectional:
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]
        
        out = self.dropout(hidden)
        out = self.fc(out)
        return out


class TransformerClassifier(nn.Module):
    """
    Transformer-based driver behavior classifier.
    
    Uses self-attention to capture long-range dependencies
    in driving behavior sequences.
    """
    
    def __init__(
        self,
        input_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use mean pooling over sequence
        x = x.mean(dim=1)
        
        out = self.dropout(x)
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DeepLearningTrainer:
    """
    Trainer for deep learning models.
    
    Handles training, validation, and evaluation of LSTM and Transformer models.
    """
    
    def __init__(
        self,
        model_type: str = 'lstm',
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,
        device: str = None
    ):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                       'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training and validation."""
        # Split by segment to avoid data leakage
        if 'segment_id' in df.columns:
            segments = df['segment_id'].unique()
            train_segs, val_segs = train_test_split(segments, test_size=0.2, random_state=42)
            train_df = df[df['segment_id'].isin(train_segs)]
            val_df = df[df['segment_id'].isin(val_segs)]
        else:
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = SequenceDataset(
            train_df, 
            sequence_length=self.sequence_length,
            fit_transforms=True
        )
        
        self.scaler = train_dataset.scaler
        self.label_encoder = train_dataset.label_encoder
        
        val_dataset = SequenceDataset(
            val_df,
            sequence_length=self.sequence_length,
            scaler=self.scaler,
            label_encoder=self.label_encoder,
            fit_transforms=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def build_model(self, input_size: int, num_classes: int) -> nn.Module:
        """Build the model architecture."""
        if self.model_type == 'lstm':
            model = LSTMClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=num_classes
            )
        elif self.model_type == 'transformer':
            model = TransformerClassifier(
                input_size=input_size,
                d_model=self.hidden_size,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the model.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(df)
        
        input_size = len([c for c in FEATURE_COLUMNS if c in df.columns])
        num_classes = len(self.label_encoder.classes_)
        
        print(f"Training sequences: {len(train_loader.dataset)}")
        print(f"Validation sequences: {len(val_loader.dataset)}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Build model
        self.model = self.build_model(input_size, num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        best_val_f1 = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1 = self._evaluate(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.3f} | "
                  f"Val F1: {val_f1:.3f}")
            
            # Track best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
        
        print(f"\nBest Validation F1: {best_val_f1:.3f}")
        return history
    
    def _evaluate(self, loader: DataLoader, criterion) -> Tuple[float, float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def save(self, path: str):
        """Save model and transforms."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, path)
        print(f"Model saved to {path}")


def train_deep_learning_models(data_path: str, output_dir: str = 'models/') -> dict:
    """
    Train both LSTM and Transformer models.
    
    Args:
        data_path: Path to training CSV
        output_dir: Directory to save models
        
    Returns:
        Results for both models including timing
    """
    import time
    
    print(f"\nLoading data from {data_path}...")
    load_start = time.time()
    df = pd.read_csv(data_path)
    load_time = time.time() - load_start
    print(f"Loaded {len(df)} samples in {load_time:.2f}s")
    
    results = {}
    timing = {'data_loading': load_time}
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    total_start = time.time()
    
    # Train LSTM
    print("\n" + "="*60)
    print("TRAINING LSTM")
    print("="*60)
    lstm_start = time.time()
    lstm_trainer = DeepLearningTrainer(
        model_type='lstm',
        sequence_length=20,
        hidden_size=64,
        epochs=15,
        batch_size=64
    )
    lstm_history = lstm_trainer.train(df)
    lstm_trainer.save(str(output_path / 'lstm_model.pt'))
    timing['lstm_training'] = time.time() - lstm_start
    results['lstm'] = {
        'best_val_f1': max(lstm_history['val_f1']),
        'best_val_acc': max(lstm_history['val_acc']),
        'training_time': timing['lstm_training']
    }
    print(f"\nLSTM Training Time: {timing['lstm_training']:.2f} seconds")
    
    # Train Transformer
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER")
    print("="*60)
    transformer_start = time.time()
    transformer_trainer = DeepLearningTrainer(
        model_type='transformer',
        sequence_length=20,
        hidden_size=64,
        epochs=15,
        batch_size=64
    )
    transformer_history = transformer_trainer.train(df)
    transformer_trainer.save(str(output_path / 'transformer_model.pt'))
    timing['transformer_training'] = time.time() - transformer_start
    results['transformer'] = {
        'best_val_f1': max(transformer_history['val_f1']),
        'best_val_acc': max(transformer_history['val_acc']),
        'training_time': timing['transformer_training']
    }
    print(f"\nTransformer Training Time: {timing['transformer_training']:.2f} seconds")
    
    timing['total'] = time.time() - total_start
    
    # Summary
    print("\n" + "="*60)
    print("DEEP LEARNING TRAINING COMPLETE")
    print("="*60)
    print(f"\n{'Model':<15} {'Best Accuracy':<15} {'Best F1':<10} {'Time (s)':<10}")
    print("-"*55)
    print(f"{'LSTM':<15} {results['lstm']['best_val_acc']:<15.3f} {results['lstm']['best_val_f1']:<10.3f} {timing['lstm_training']:<10.2f}")
    print(f"{'Transformer':<15} {results['transformer']['best_val_acc']:<15.3f} {results['transformer']['best_val_f1']:<10.3f} {timing['transformer_training']:<10.2f}")
    
    print("\n--- Timing Summary ---")
    print(f"  Data Loading:          {timing['data_loading']:.2f}s")
    print(f"  LSTM Training:         {timing['lstm_training']:.2f}s")
    print(f"  Transformer Training:  {timing['transformer_training']:.2f}s")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:                 {timing['total']:.2f}s")
    
    results['timing'] = timing
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deep learning models")
    parser.add_argument("--train", type=str, default="data/processed/training_data.csv",
                       help="Path to training data CSV")
    parser.add_argument("--output", "-o", type=str, default="models/",
                       help="Output directory for models")
    parser.add_argument("--model", type=str, choices=['lstm', 'transformer', 'both'],
                       default='both', help="Which model(s) to train")
    
    args = parser.parse_args()
    
    if args.model == 'both':
        results = train_deep_learning_models(args.train, args.output)
    else:
        df = pd.read_csv(args.train)
        trainer = DeepLearningTrainer(model_type=args.model, epochs=15)
        trainer.train(df)
        trainer.save(f"{args.output}/{args.model}_model.pt")
