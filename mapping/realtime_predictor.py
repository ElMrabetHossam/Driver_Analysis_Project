"""
Real-time Prediction Module
Loads trained Transformer model and makes frame-by-frame predictions
with intelligent alerts for dangerous driving behavior
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# Feature columns that the model expects
FEATURE_COLUMNS = [
    'speed', 'steering', 'accel_forward', 'accel_lateral', 
    'gyro_yaw', 'radar_distance', 'vehicle_count',
    'speed_change', 'steering_rate', 'steering_jerk'
]

# Alert thresholds and configurations
ALERT_CONFIG = {
    'aggressive': {
        'color': '#ff0000',
        'icon': '🔴',
        'priority': 3,
        'message': 'ALERTE DANGER: Conduite Agressive Détectée!'
    },
    'moderate': {
        'color': '#ffa500',
        'icon': '⚠️',
        'priority': 2,
        'message': 'Attention: Conduite Modérée'
    },
    'safe': {
        'color': '#00ff00',
        'icon': '✅',
        'priority': 1,
        'message': 'Conduite Sécurisée'
    }
}


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


class TransformerClassifier(nn.Module):
    """Transformer-based driver behavior classifier."""
    
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


class RealtimePredictor:
    """
    Real-time prediction engine for driver behavior.
    
    Maintains a sliding window of frames and makes predictions
    with intelligent alerting system.
    """
    
    def __init__(self, model_path: str, sequence_length: int = 20, device: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved transformer model (.pt file)
            sequence_length: Number of frames to use for prediction
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.sequence_length = sequence_length
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        # We need weights_only=False because the checkpoint contains sklearn objects
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract parameters
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        hidden_size = checkpoint.get('hidden_size', 64)
        
        # Determine input size from scaler
        input_size = self.scaler.n_features_in_
        num_classes = len(self.label_encoder.classes_)
        
        # Build model
        self.model = TransformerClassifier(
            input_size=input_size,
            d_model=hidden_size,
            num_classes=num_classes
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Sliding window buffer
        self.feature_buffer = deque(maxlen=sequence_length)
        
        # Alert tracking
        self.alert_history = deque(maxlen=100)
        self.current_alert = None
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Sequence Length: {sequence_length}")
    
    def extract_features(self, frame_data: dict, radar_targets: list, wheel_data: dict) -> np.ndarray:
        """
        Extract features from frame data for prediction.
        
        Args:
            frame_data: Dict with telemetry data
            radar_targets: List of radar targets
            wheel_data: Dict with wheel dynamics
            
        Returns:
            Feature vector as numpy array
        """
        features = {}
        
        # Speed (km/h -> m/s)
        features['speed'] = frame_data.get('longitudinal_velocity', 0)
        
        # Steering angle
        steering = frame_data.get('steering', 0)
        if hasattr(steering, '__len__'):
            steering = float(steering[0])
        features['steering'] = float(steering) if steering is not None else 0.0
        
        # Accelerations
        if 'imu_accel' in frame_data and frame_data['imu_accel'] is not None:
            ax, ay, az = frame_data['imu_accel']
            features['accel_forward'] = ax
            features['accel_lateral'] = ay
        else:
            features['accel_forward'] = 0.0
            features['accel_lateral'] = 0.0
        
        # Gyro (yaw rate)
        if 'gyro' in frame_data and frame_data['gyro'] is not None:
            try:
                gz = float(frame_data['gyro'][2])
                features['gyro_yaw'] = gz
            except:
                features['gyro_yaw'] = 0.0
        else:
            features['gyro_yaw'] = 0.0
        
        # Radar distance (use closest target)
        if radar_targets and len(radar_targets) > 0:
            min_dist = min([t.get('dist', 999) for t in radar_targets])
            features['radar_distance'] = min_dist if min_dist < 999 else 0.0
        else:
            features['radar_distance'] = 0.0
        
        # Vehicle count (placeholder - would come from YOLO detections)
        features['vehicle_count'] = len(radar_targets) if radar_targets else 0
        
        # Derived features (calculate from history if available)
        if len(self.feature_buffer) > 0:
            prev_features = self.feature_buffer[-1]
            features['speed_change'] = features['speed'] - prev_features[0]
            features['steering_rate'] = features['steering'] - prev_features[1]
            
            # Steering jerk (second derivative)
            if len(self.feature_buffer) > 1:
                prev_prev_steering = self.feature_buffer[-2][1]
                prev_steering_rate = prev_features[1] - prev_prev_steering
                features['steering_jerk'] = features['steering_rate'] - prev_steering_rate
            else:
                features['steering_jerk'] = 0.0
        else:
            features['speed_change'] = 0.0
            features['steering_rate'] = 0.0
            features['steering_jerk'] = 0.0
        
        # Convert to array in correct order
        feature_array = np.array([
            features.get(col, 0.0) for col in FEATURE_COLUMNS
        ])
        
        return feature_array
    
    def predict(self, frame_data: dict, radar_targets: list = None, 
                wheel_data: dict = None) -> dict:
        """
        Make prediction for current frame.
        
        Args:
            frame_data: Frame telemetry data
            radar_targets: Radar targets (optional)
            wheel_data: Wheel dynamics data (optional)
            
        Returns:
            Dict with prediction results and alert information
        """
        # Extract features
        features = self.extract_features(
            frame_data, 
            radar_targets or [],
            wheel_data or {}
        )
        
        # Add to buffer
        self.feature_buffer.append(features)
        
        # Need full sequence for prediction
        if len(self.feature_buffer) < self.sequence_length:
            return {
                'prediction': 'safe',
                'confidence': 0.0,
                'probabilities': {'safe': 1.0, 'moderate': 0.0, 'aggressive': 0.0},
                'alert': None,
                'ready': False
            }
        
        # Prepare sequence
        sequence = np.array(list(self.feature_buffer))
        
        # Handle inf/nan
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        sequence_scaled = self.scaler.transform(sequence)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction_idx = outputs.argmax(dim=1).item()
        
        # Decode prediction
        prediction_label = self.label_encoder.inverse_transform([prediction_idx])[0]
        probs_dict = {
            label: float(probabilities[i].cpu().numpy())
            for i, label in enumerate(self.label_encoder.classes_)
        }
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[prediction_idx].cpu().numpy())
        
        # Generate alert
        alert = self._generate_alert(prediction_label, confidence, probs_dict)
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': probs_dict,
            'alert': alert,
            'ready': True
        }
    
    def _generate_alert(self, prediction: str, confidence: float, probabilities: dict) -> dict:
        """
        Generate alert based on prediction.
        
        Args:
            prediction: Predicted class
            confidence: Confidence score
            probabilities: Class probabilities
            
        Returns:
            Alert dictionary
        """
        # Get alert config for prediction
        alert_config = ALERT_CONFIG.get(prediction, ALERT_CONFIG['safe'])
        
        # Adjust message based on confidence
        if confidence > 0.8:
            severity = "CRITIQUE"
        elif confidence > 0.6:
            severity = "ÉLEVÉ"
        elif confidence > 0.4:
            severity = "MOYEN"
        else:
            severity = "FAIBLE"
        
        # Build alert
        alert = {
            'type': prediction,
            'severity': severity,
            'confidence': confidence,
            'color': alert_config['color'],
            'icon': alert_config['icon'],
            'priority': alert_config['priority'],
            'message': alert_config['message'],
            'probabilities': probabilities,
            'timestamp': len(self.alert_history)
        }
        
        # Track alert
        self.alert_history.append(alert)
        self.current_alert = alert
        
        return alert
    
    def get_alert_summary(self, last_n: int = 10) -> dict:
        """
        Get summary of recent alerts.
        
        Args:
            last_n: Number of recent alerts to summarize
            
        Returns:
            Summary statistics
        """
        recent_alerts = list(self.alert_history)[-last_n:]
        
        if not recent_alerts:
            return {
                'total': 0,
                'aggressive_count': 0,
                'moderate_count': 0,
                'safe_count': 0
            }
        
        summary = {
            'total': len(recent_alerts),
            'aggressive_count': sum(1 for a in recent_alerts if a['type'] == 'aggressive'),
            'moderate_count': sum(1 for a in recent_alerts if a['type'] == 'moderate'),
            'safe_count': sum(1 for a in recent_alerts if a['type'] == 'safe'),
            'avg_confidence': np.mean([a['confidence'] for a in recent_alerts])
        }
        
        return summary


# Singleton instance
_predictor_instance = None

def get_predictor(model_path: str = None, **kwargs):
    """Get or create predictor instance."""
    global _predictor_instance
    if _predictor_instance is None and model_path:
        _predictor_instance = RealtimePredictor(model_path, **kwargs)
    return _predictor_instance
