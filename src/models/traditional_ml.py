"""
Traditional Machine Learning Models for Driver Behavior Classification

This module implements:
- SVM (Support Vector Machine) - Classification
- Random Forest - Classification
- K-Means - Clustering driver styles
- Isolation Forest - Anomaly detection

Usage:
    python -m src.models.traditional_ml --train data/processed/training_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, silhouette_score
)


# Features to use for training (exclude metadata columns)
FEATURE_COLUMNS = [
    'speed', 'steering', 'accel_forward', 'accel_lateral', 
    'gyro_yaw', 'radar_distance', 'vehicle_count',
    'speed_change', 'steering_rate', 'steering_jerk'
]


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    classification_report: str


class DriverBehaviorClassifier:
    """
    Multi-model classifier for driver behavior.
    
    Supports SVM, Random Forest, and ensemble predictions.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier.
        
        Args:
            model_type: 'svm', 'random_forest', or 'ensemble'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_fitted = False
        
        # Model-specific settings
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', 
                           class_weight='balanced', random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=15, 
                class_weight='balanced', random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from dataframe.
        
        Args:
            df: DataFrame with features and 'label' column
            
        Returns:
            (X, y) tuple of features and encoded labels
        """
        # Select feature columns that exist
        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Get labels
        y = df['label'].values
        
        return X.values, y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DriverBehaviorClassifier':
        """
        Train the model.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            
        Returns:
            self
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new data.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels (string)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelResults:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            ModelResults with metrics
        """
        y_pred = self.predict(X)
        y_encoded = self.label_encoder.transform(y)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        return ModelResults(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y_encoded, y_pred_encoded, average='weighted'),
            recall=recall_score(y_encoded, y_pred_encoded, average='weighted'),
            f1=f1_score(y_encoded, y_pred_encoded, average='weighted'),
            confusion_matrix=confusion_matrix(y, y_pred),
            classification_report=classification_report(y, y_pred)
        )
    
    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance (Random Forest only)."""
        if self.model_type != 'random_forest':
            return None
        
        available_features = [c for c in FEATURE_COLUMNS][:len(self.model.feature_importances_)]
        importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'DriverBehaviorClassifier':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.label_encoder = data['label_encoder']
        classifier.is_fitted = data['is_fitted']
        
        return classifier


class DrivingStyleClusterer:
    """
    K-Means clustering to discover driving styles.
    
    Useful for unsupervised analysis of driver behavior patterns.
    """
    
    def __init__(self, n_clusters: int = 3):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of driving style clusters
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'DrivingStyleClusterer':
        """Fit clustering model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_cluster_centers(self) -> pd.DataFrame:
        """Get cluster center characteristics."""
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        feature_names = FEATURE_COLUMNS[:centers.shape[1]]
        
        df = pd.DataFrame(centers, columns=feature_names)
        df.index = [f'Cluster_{i}' for i in range(self.n_clusters)]
        return df
    
    def evaluate(self, X: np.ndarray) -> Dict:
        """Evaluate clustering quality."""
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        
        return {
            'n_clusters': self.n_clusters,
            'inertia': self.model.inertia_,
            'silhouette_score': silhouette_score(X_scaled, labels),
            'cluster_sizes': np.bincount(labels)
        }


class AnomalyDetector:
    """
    Isolation Forest for detecting dangerous driving anomalies.
    
    Identifies unusual driving patterns that may indicate danger.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0-0.5)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        """Fit anomaly detector."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Returns:
            Array of 1 (normal) or -1 (anomaly)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.
        
        Lower (more negative) = more anomalous.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray = None) -> Dict:
        """
        Evaluate anomaly detection.
        
        Args:
            X: Features
            y_true: Optional true labels for comparison
        """
        predictions = self.predict(X)
        anomalies = predictions == -1
        
        results = {
            'total_samples': len(X),
            'anomalies_detected': anomalies.sum(),
            'anomaly_rate': anomalies.mean()
        }
        
        # If true labels provided, check correlation
        if y_true is not None:
            # Assume 'aggressive' should correlate with anomalies
            is_dangerous = np.isin(y_true, ['aggressive'])
            
            # Precision: of detected anomalies, how many are actually dangerous?
            if anomalies.sum() > 0:
                results['precision'] = (anomalies & is_dangerous).sum() / anomalies.sum()
            
            # Recall: of actual dangerous, how many did we detect?
            if is_dangerous.sum() > 0:
                results['recall'] = (anomalies & is_dangerous).sum() / is_dangerous.sum()
        
        return results


def train_all_models(data_path: str, output_dir: str = 'models/') -> Dict:
    """
    Train all ML models on the dataset.
    
    Args:
        data_path: Path to training CSV
        output_dir: Directory to save models
        
    Returns:
        Dict with all model results
    """
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Prepare features
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].fillna(df[available_features].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    y = df['label'].values
    
    print(f"Features: {available_features}")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. SVM
    print("\n" + "-"*40)
    print("Training SVM...")
    svm = DriverBehaviorClassifier(model_type='svm')
    svm.fit(X_train, y_train)
    svm_results = svm.evaluate(X_test, y_test)
    results['svm'] = svm_results
    svm.save(str(output_path / 'svm_model.pkl'))
    print(f"SVM Accuracy: {svm_results.accuracy:.3f}")
    print(f"SVM F1 Score: {svm_results.f1:.3f}")
    
    # 2. Random Forest
    print("\n" + "-"*40)
    print("Training Random Forest...")
    rf = DriverBehaviorClassifier(model_type='random_forest')
    rf.fit(X_train, y_train)
    rf_results = rf.evaluate(X_test, y_test)
    results['random_forest'] = rf_results
    rf.save(str(output_path / 'random_forest_model.pkl'))
    print(f"Random Forest Accuracy: {rf_results.accuracy:.3f}")
    print(f"Random Forest F1 Score: {rf_results.f1:.3f}")
    
    # Feature importance
    importance = rf.feature_importance()
    if importance is not None:
        print("\nFeature Importance:")
        for _, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # 3. K-Means Clustering
    print("\n" + "-"*40)
    print("Training K-Means Clustering...")
    kmeans = DrivingStyleClusterer(n_clusters=3)
    kmeans.fit(X_train)
    kmeans_eval = kmeans.evaluate(X_train)
    results['kmeans'] = kmeans_eval
    print(f"Silhouette Score: {kmeans_eval['silhouette_score']:.3f}")
    print(f"Cluster sizes: {kmeans_eval['cluster_sizes']}")
    
    # 4. Isolation Forest
    print("\n" + "-"*40)
    print("Training Isolation Forest...")
    iso = AnomalyDetector(contamination=0.1)
    iso.fit(X_train)
    iso_eval = iso.evaluate(X_test, y_test)
    results['isolation_forest'] = iso_eval
    print(f"Anomalies detected: {iso_eval['anomalies_detected']}/{iso_eval['total_samples']}")
    if 'precision' in iso_eval:
        print(f"Precision (anomaly=aggressive): {iso_eval.get('precision', 0):.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {output_path}/")
    print(f"  - svm_model.pkl")
    print(f"  - random_forest_model.pkl")
    
    print("\n--- Classification Results ---")
    print(f"{'Model':<15} {'Accuracy':<10} {'F1 Score':<10}")
    print("-"*35)
    print(f"{'SVM':<15} {results['svm'].accuracy:<10.3f} {results['svm'].f1:<10.3f}")
    print(f"{'Random Forest':<15} {results['random_forest'].accuracy:<10.3f} {results['random_forest'].f1:<10.3f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--train", type=str, default="data/processed/training_data.csv",
                       help="Path to training data CSV")
    parser.add_argument("--output", "-o", type=str, default="models/",
                       help="Output directory for models")
    
    args = parser.parse_args()
    
    results = train_all_models(args.train, args.output)
