"""
Driving Behavior Labeler

Labels driving frames as 'safe', 'aggressive', or 'drowsy' based on
heuristic rules applied to sensor and visual features.

These labels are used for supervised learning (SVM, Random Forest, LSTM, Transformer).

Labeling rules:
- AGGRESSIVE: High speed, sudden steering, tailgating, hard braking
- DROWSY: Excessive lane deviation, inconsistent steering
- SAFE: Normal driving within safe parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LabelThresholds:
    """
    Configurable thresholds for behavior labeling.
    
    These values should be tuned based on the dataset characteristics
    and domain knowledge.
    """
    # Speed thresholds
    high_speed: float = 30.0           # m/s (~108 km/h, ~67 mph) - highway speeding
    very_high_speed: float = 35.0      # m/s (~126 km/h, ~78 mph) - dangerous speeding
    
    # Steering thresholds
    high_steering_rate: float = 15.0   # deg/s - rapid steering changes
    high_steering_jerk: float = 30.0   # deg/s^2 - sudden steering adjustments
    
    # Following distance
    tailgating_distance: float = 15.0  # meters - unsafe following at highway speed
    safe_time_headway: float = 2.0     # seconds - 2-second rule
    
    # Lane keeping
    lane_deviation_warning: float = 0.3  # meters - drifting
    lane_deviation_danger: float = 0.5   # meters - significant drift
    
    # Acceleration
    hard_braking: float = -3.0         # m/s^2 - hard braking event
    hard_acceleration: float = 3.0     # m/s^2 - aggressive acceleration
    
    # Consistency (rolling window std)
    inconsistent_steering_threshold: float = 5.0  # degrees - could indicate drowsiness


class DrivingLabeler:
    """
    Label driving behavior based on heuristic rules.
    
    Usage:
        labeler = DrivingLabeler()
        df['label'] = labeler.label_dataframe(df)
    """
    
    # Label constants
    SAFE = 'safe'
    AGGRESSIVE = 'aggressive'
    DROWSY = 'drowsy'
    
    def __init__(self, thresholds: LabelThresholds = None):
        """
        Initialize the labeler.
        
        Args:
            thresholds: Custom thresholds, or None for defaults
        """
        self.thresholds = thresholds or LabelThresholds()
    
    def _check_aggressive(self, features: Dict) -> Tuple[bool, List[str]]:
        """
        Check for aggressive driving indicators.
        
        Args:
            features: Dict of feature values for a single frame
            
        Returns:
            Tuple of (is_aggressive, list_of_reasons)
        """
        reasons = []
        t = self.thresholds
        
        # High speed
        speed = features.get('speed', 0)
        if speed > t.very_high_speed:
            reasons.append(f'very_high_speed ({speed:.1f} m/s)')
        elif speed > t.high_speed:
            reasons.append(f'high_speed ({speed:.1f} m/s)')
        
        # Rapid steering
        steering_rate = abs(features.get('steering_rate', 0))
        if steering_rate > t.high_steering_rate:
            reasons.append(f'rapid_steering ({steering_rate:.1f} deg/s)')
        
        # Sudden steering (jerk)
        steering_jerk = abs(features.get('steering_jerk', 0))
        if steering_jerk > t.high_steering_jerk:
            reasons.append(f'sudden_steering ({steering_jerk:.1f} deg/s^2)')
        
        # Tailgating - check both radar and visual distance
        radar_dist = features.get('radar_distance', np.nan)
        visual_dist = features.get('lead_distance_visual', np.nan)
        lead_dist = radar_dist if not np.isnan(radar_dist) else visual_dist
        
        if not np.isnan(lead_dist) and lead_dist < t.tailgating_distance and speed > 10:
            # Calculate time headway
            time_headway = lead_dist / max(speed, 0.1)
            if time_headway < t.safe_time_headway:
                reasons.append(f'tailgating ({lead_dist:.1f}m, {time_headway:.1f}s headway)')
        
        # Hard braking
        accel = features.get('accel_forward', 0)
        if accel < t.hard_braking:
            reasons.append(f'hard_braking ({accel:.1f} m/s^2)')
        
        # Hard acceleration
        if accel > t.hard_acceleration:
            reasons.append(f'hard_acceleration ({accel:.1f} m/s^2)')
        
        return len(reasons) > 0, reasons
    
    def _check_drowsy(self, features: Dict) -> Tuple[bool, List[str]]:
        """
        Check for drowsy driving indicators.
        
        Args:
            features: Dict of feature values for a single frame
            
        Returns:
            Tuple of (is_drowsy, list_of_reasons)
        """
        reasons = []
        t = self.thresholds
        
        # Lane deviation
        lane_dev = abs(features.get('lane_deviation', np.nan))
        if not np.isnan(lane_dev):
            if lane_dev > t.lane_deviation_danger:
                reasons.append(f'lane_deviation_danger ({lane_dev:.2f}m)')
            elif lane_dev > t.lane_deviation_warning:
                reasons.append(f'lane_deviation_warning ({lane_dev:.2f}m)')
        
        # Note: For more robust drowsiness detection, we would analyze
        # patterns over a time window (e.g., micro-corrections, drift patterns)
        # This is handled in label_with_context()
        
        return len(reasons) > 0, reasons
    
    def label_frame(self, features: Dict) -> str:
        """
        Label a single frame based on its features.
        
        Args:
            features: Dict of feature values
            
        Returns:
            Label string: 'safe', 'aggressive', or 'drowsy'
        """
        is_aggressive, _ = self._check_aggressive(features)
        if is_aggressive:
            return self.AGGRESSIVE
        
        is_drowsy, _ = self._check_drowsy(features)
        if is_drowsy:
            return self.DROWSY
        
        return self.SAFE
    
    def label_frame_detailed(self, features: Dict) -> Tuple[str, List[str]]:
        """
        Label a frame with detailed reasons.
        
        Args:
            features: Dict of feature values
            
        Returns:
            Tuple of (label, list_of_reasons)
        """
        is_aggressive, aggressive_reasons = self._check_aggressive(features)
        if is_aggressive:
            return self.AGGRESSIVE, aggressive_reasons
        
        is_drowsy, drowsy_reasons = self._check_drowsy(features)
        if is_drowsy:
            return self.DROWSY, drowsy_reasons
        
        return self.SAFE, []
    
    def label_dataframe(self, df: pd.DataFrame, detailed: bool = False) -> pd.Series:
        """
        Label an entire DataFrame.
        
        Args:
            df: DataFrame with feature columns
            detailed: If True, also add 'label_reasons' column
            
        Returns:
            Series of labels (and modifies df in place if detailed=True)
        """
        labels = []
        reasons_list = []
        
        for idx, row in df.iterrows():
            features = row.to_dict()
            if detailed:
                label, reasons = self.label_frame_detailed(features)
                labels.append(label)
                reasons_list.append('; '.join(reasons) if reasons else '')
            else:
                labels.append(self.label_frame(features))
        
        if detailed:
            df['label_reasons'] = reasons_list
        
        return pd.Series(labels, index=df.index)
    
    def label_with_context(self, 
                           df: pd.DataFrame, 
                           window_size: int = 50) -> pd.Series:
        """
        Label with temporal context using rolling windows.
        
        This enables detection of patterns like:
        - Drowsy drifting (alternating lane deviations)
        - Inconsistent steering (high variance over time)
        
        Args:
            df: DataFrame with feature columns
            window_size: Number of frames for rolling statistics
            
        Returns:
            Series of labels
        """
        labels = self.label_dataframe(df).copy()
        t = self.thresholds
        
        # Calculate rolling statistics for drowsiness detection
        if 'steering' in df.columns:
            steering_rolling_std = df['steering'].rolling(window=window_size, min_periods=10).std()
            
            # Mark as drowsy if high steering variance but not aggressive
            drowsy_mask = (
                (steering_rolling_std > t.inconsistent_steering_threshold) & 
                (labels == self.SAFE)
            )
            labels[drowsy_mask] = self.DROWSY
        
        if 'lane_deviation' in df.columns:
            # Check for alternating lane drifts (sign changes)
            lane_dev = df['lane_deviation'].fillna(0)
            sign_changes = (np.sign(lane_dev).diff().fillna(0) != 0).rolling(
                window=window_size, min_periods=10
            ).sum()
            
            # Many sign changes could indicate drowsy weaving
            weaving_mask = (sign_changes > window_size * 0.3) & (labels == self.SAFE)
            labels[weaving_mask] = self.DROWSY
        
        return labels
    
    def get_label_statistics(self, labels: pd.Series) -> Dict:
        """
        Get statistics about the label distribution.
        
        Args:
            labels: Series of labels
            
        Returns:
            Dict with counts and percentages
        """
        counts = labels.value_counts()
        total = len(labels)
        
        stats = {
            'total_frames': total,
            'safe_count': counts.get(self.SAFE, 0),
            'aggressive_count': counts.get(self.AGGRESSIVE, 0),
            'drowsy_count': counts.get(self.DROWSY, 0),
            'safe_pct': counts.get(self.SAFE, 0) / total * 100,
            'aggressive_pct': counts.get(self.AGGRESSIVE, 0) / total * 100,
            'drowsy_pct': counts.get(self.DROWSY, 0) / total * 100,
        }
        
        return stats
    
    def create_binary_labels(self, labels: pd.Series) -> pd.Series:
        """
        Convert multi-class labels to binary (safe vs dangerous).
        
        Args:
            labels: Series of 'safe', 'aggressive', 'drowsy' labels
            
        Returns:
            Series of 0 (safe) or 1 (dangerous)
        """
        return (labels != self.SAFE).astype(int)


def label_csv_file(input_path: str, output_path: str, use_context: bool = True):
    """
    Load a feature CSV, add labels, and save.
    
    Args:
        input_path: Path to input CSV with features
        output_path: Path to save labeled CSV
        use_context: Whether to use temporal context for labeling
    """
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Labeling frames...")
    labeler = DrivingLabeler()
    
    if use_context:
        df['label'] = labeler.label_with_context(df)
    else:
        df['label'] = labeler.label_dataframe(df)
    
    # Add binary label
    df['is_dangerous'] = labeler.create_binary_labels(df['label'])
    
    # Print statistics
    stats = labeler.get_label_statistics(df['label'])
    print(f"\nLabel Statistics:")
    print(f"  Safe: {stats['safe_count']} ({stats['safe_pct']:.1f}%)")
    print(f"  Aggressive: {stats['aggressive_count']} ({stats['aggressive_pct']:.1f}%)")
    print(f"  Drowsy: {stats['drowsy_count']} ({stats['drowsy_pct']:.1f}%)")
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nLabeled data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Label driving behavior")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, help="Output CSV path (default: add _labeled suffix)")
    parser.add_argument("--no-context", action="store_true", help="Disable temporal context")
    
    args = parser.parse_args()
    
    output_path = args.output
    if not output_path:
        from pathlib import Path
        p = Path(args.input)
        output_path = str(p.parent / f"{p.stem}_labeled{p.suffix}")
    
    label_csv_file(args.input, output_path, use_context=not args.no_context)
