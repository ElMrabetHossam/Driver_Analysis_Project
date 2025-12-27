"""
Driver Safety Scoring Algorithm

Computes a driver safety score (0-100) based on:
- ML model predictions (aggressive/safe/drowsy classification)
- Real-time driving metrics (speed, steering jerk, following distance)
- Historical behavior patterns

Usage:
    from src.models.scorer import DriverScorer
    scorer = DriverScorer()
    score = scorer.compute_score(features_df)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import pickle


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of driver safety score."""
    overall_score: float  # 0-100
    behavior_score: float  # Based on ML classification
    smoothness_score: float  # Based on steering/speed consistency
    awareness_score: float  # Based on following distance
    speed_score: float  # Based on speed profile
    risk_factors: list  # List of identified risk factors
    recommendations: list  # Safety recommendations


class DriverScorer:
    """
    Compute driver safety score from driving features.
    
    Combines ML predictions with physics-based metrics for
    a comprehensive safety assessment.
    """
    
    # Score weights
    WEIGHTS = {
        'behavior': 0.40,  # ML classification
        'smoothness': 0.25,  # Steering/acceleration smoothness
        'awareness': 0.20,  # Following distance
        'speed': 0.15  # Speed compliance
    }
    
    # Thresholds for scoring
    THRESHOLDS = {
        'speed_limit_ms': 35.0,  # ~126 km/h highway limit
        'min_following_distance': 30.0,  # meters
        'max_steering_jerk': 5.0,  # degrees/s^2
        'max_speed_change': 3.0,  # m/s^2
        'max_steering_rate': 10.0  # degrees/s
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize scorer.
        
        Args:
            model_path: Path to trained Random Forest model (optional)
        """
        self.model = None
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        elif Path('models/random_forest_model.pkl').exists():
            self._load_model('models/random_forest_model.pkl')
    
    def _load_model(self, path: str):
        """Load trained classifier for behavior scoring."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.model = data.get('model')
            self.scaler = data.get('scaler')
            self.label_encoder = data.get('label_encoder')
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
    
    def compute_score(self, df: pd.DataFrame) -> ScoreBreakdown:
        """
        Compute overall driver safety score.
        
        Args:
            df: DataFrame with driving features
            
        Returns:
            ScoreBreakdown with detailed scoring information
        """
        # Compute component scores
        behavior_score = self._compute_behavior_score(df)
        smoothness_score = self._compute_smoothness_score(df)
        awareness_score = self._compute_awareness_score(df)
        speed_score = self._compute_speed_score(df)
        
        # Weighted average
        overall_score = (
            self.WEIGHTS['behavior'] * behavior_score +
            self.WEIGHTS['smoothness'] * smoothness_score +
            self.WEIGHTS['awareness'] * awareness_score +
            self.WEIGHTS['speed'] * speed_score
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            behavior_score, smoothness_score, awareness_score, speed_score, risk_factors
        )
        
        return ScoreBreakdown(
            overall_score=round(overall_score, 1),
            behavior_score=round(behavior_score, 1),
            smoothness_score=round(smoothness_score, 1),
            awareness_score=round(awareness_score, 1),
            speed_score=round(speed_score, 1),
            risk_factors=risk_factors,
            recommendations=recommendations
        )
    
    def _compute_behavior_score(self, df: pd.DataFrame) -> float:
        """Score based on ML behavior classification."""
        if 'label' in df.columns:
            # Use existing labels
            label_counts = df['label'].value_counts(normalize=True)
            safe_ratio = label_counts.get('safe', 0)
            drowsy_ratio = label_counts.get('drowsy', 0)
            aggressive_ratio = label_counts.get('aggressive', 0)
            
            # Safe = 100 pts, Drowsy = 40 pts, Aggressive = 20 pts
            score = safe_ratio * 100 + drowsy_ratio * 40 + aggressive_ratio * 20
            return min(100, max(0, score))
        
        elif self.model is not None:
            # Use ML model to predict
            feature_cols = ['speed', 'steering', 'accel_forward', 'accel_lateral',
                          'gyro_yaw', 'radar_distance', 'vehicle_count',
                          'speed_change', 'steering_rate', 'steering_jerk']
            available = [c for c in feature_cols if c in df.columns]
            X = df[available].fillna(0).values
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            labels = self.label_encoder.inverse_transform(predictions)
            
            safe_ratio = (labels == 'safe').mean()
            return min(100, max(0, safe_ratio * 100))
        
        return 50.0  # Default if no model/labels
    
    def _compute_smoothness_score(self, df: pd.DataFrame) -> float:
        """Score based on driving smoothness (low jerk, consistent steering)."""
        scores = []
        
        # Steering jerk penalty
        if 'steering_jerk' in df.columns:
            jerk = df['steering_jerk'].abs()
            jerk_score = 100 - min(100, (jerk.mean() / self.THRESHOLDS['max_steering_jerk']) * 50)
            scores.append(jerk_score)
        
        # Steering rate consistency
        if 'steering_rate' in df.columns:
            rate = df['steering_rate'].abs()
            rate_score = 100 - min(100, (rate.mean() / self.THRESHOLDS['max_steering_rate']) * 50)
            scores.append(rate_score)
        
        # Speed change (acceleration) smoothness
        if 'speed_change' in df.columns:
            accel = df['speed_change'].abs()
            accel_score = 100 - min(100, (accel.mean() / self.THRESHOLDS['max_speed_change']) * 50)
            scores.append(accel_score)
        
        return np.mean(scores) if scores else 50.0
    
    def _compute_awareness_score(self, df: pd.DataFrame) -> float:
        """Score based on following distance and situational awareness."""
        scores = []
        
        # Following distance from radar
        if 'radar_distance' in df.columns:
            valid_distances = df['radar_distance'].dropna()
            if len(valid_distances) > 0:
                min_threshold = self.THRESHOLDS['min_following_distance']
                
                # Penalize close following
                close_ratio = (valid_distances < min_threshold).mean()
                distance_score = 100 - (close_ratio * 100)
                scores.append(distance_score)
                
                # Bonus for maintaining good distance
                avg_distance = valid_distances.mean()
                if avg_distance > min_threshold * 2:
                    scores.append(100)
                elif avg_distance > min_threshold:
                    scores.append(80)
                else:
                    scores.append(50)
        
        # Vehicle awareness (detecting vehicles)
        if 'vehicle_count' in df.columns:
            # Consistent detection is good
            detection_rate = (df['vehicle_count'] > 0).mean()
            scores.append(50 + detection_rate * 50)
        
        return np.mean(scores) if scores else 50.0
    
    def _compute_speed_score(self, df: pd.DataFrame) -> float:
        """Score based on speed compliance."""
        if 'speed' not in df.columns:
            return 50.0
        
        speed = df['speed']
        limit = self.THRESHOLDS['speed_limit_ms']
        
        # Speeding penalty
        speeding_ratio = (speed > limit).mean()
        if speeding_ratio > 0.5:
            return 20.0
        elif speeding_ratio > 0.2:
            return 50.0
        elif speeding_ratio > 0.05:
            return 75.0
        else:
            return 100.0
    
    def _identify_risk_factors(self, df: pd.DataFrame) -> list:
        """Identify specific risk factors from driving data."""
        risks = []
        
        # High steering jerk
        if 'steering_jerk' in df.columns:
            high_jerk = (df['steering_jerk'].abs() > self.THRESHOLDS['max_steering_jerk']).mean()
            if high_jerk > 0.2:
                risks.append(f"Frequent abrupt steering ({high_jerk*100:.0f}% of time)")
        
        # Close following
        if 'radar_distance' in df.columns:
            valid = df['radar_distance'].dropna()
            if len(valid) > 0:
                close = (valid < self.THRESHOLDS['min_following_distance']).mean()
                if close > 0.3:
                    risks.append(f"Tailgating detected ({close*100:.0f}% of time)")
        
        # Aggressive acceleration
        if 'speed_change' in df.columns:
            hard_accel = (df['speed_change'].abs() > self.THRESHOLDS['max_speed_change']).mean()
            if hard_accel > 0.2:
                risks.append(f"Hard acceleration/braking ({hard_accel*100:.0f}% of time)")
        
        # Speeding
        if 'speed' in df.columns:
            speeding = (df['speed'] > self.THRESHOLDS['speed_limit_ms']).mean()
            if speeding > 0.1:
                risks.append(f"Speeding ({speeding*100:.0f}% of time)")
        
        return risks
    
    def _generate_recommendations(
        self,
        behavior_score: float,
        smoothness_score: float,
        awareness_score: float,
        speed_score: float,
        risk_factors: list
    ) -> list:
        """Generate personalized safety recommendations."""
        recommendations = []
        
        if behavior_score < 60:
            recommendations.append("Take a break - signs of aggressive or fatigued driving detected")
        
        if smoothness_score < 60:
            recommendations.append("Practice smoother steering and acceleration for better vehicle control")
        
        if awareness_score < 60:
            recommendations.append("Increase following distance - maintain at least 3-second gap")
        
        if speed_score < 60:
            recommendations.append("Reduce speed to stay within safe limits")
        
        if not recommendations:
            recommendations.append("Great driving! Maintain current safe driving habits")
        
        return recommendations
    
    def score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def score_to_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level."""
        if score >= 80:
            return 'LOW RISK'
        elif score >= 60:
            return 'MODERATE RISK'
        elif score >= 40:
            return 'HIGH RISK'
        else:
            return 'CRITICAL RISK'


def score_csv_file(csv_path: str) -> ScoreBreakdown:
    """
    Score a driving session from a CSV file.
    
    Args:
        csv_path: Path to features CSV
        
    Returns:
        ScoreBreakdown with safety assessment
    """
    df = pd.read_csv(csv_path)
    scorer = DriverScorer()
    return scorer.compute_score(df)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Score driver safety")
    parser.add_argument("--input", "-i", type=str, 
                       default="data/processed/training_data_150.csv",
                       help="Input CSV with driving features")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DRIVER SAFETY SCORE ANALYSIS")
    print("="*60)
    
    result = score_csv_file(args.input)
    
    print(f"\n{'='*40}")
    print(f"  OVERALL SCORE: {result.overall_score}/100")
    print(f"  GRADE: {DriverScorer().score_to_grade(result.overall_score)}")
    print(f"  RISK: {DriverScorer().score_to_risk_level(result.overall_score)}")
    print(f"{'='*40}")
    
    print("\n--- Score Breakdown ---")
    print(f"  Behavior:   {result.behavior_score}/100")
    print(f"  Smoothness: {result.smoothness_score}/100")
    print(f"  Awareness:  {result.awareness_score}/100")
    print(f"  Speed:      {result.speed_score}/100")
    
    if result.risk_factors:
        print("\n--- Risk Factors ---")
        for risk in result.risk_factors:
            print(f"  ⚠️  {risk}")
    
    print("\n--- Recommendations ---")
    for rec in result.recommendations:
        print(f"  → {rec}")
