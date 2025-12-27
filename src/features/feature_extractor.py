"""
Feature Extractor

Extracts features from video frames and sensor data, combining them into
a unified feature vector for machine learning.

Features extracted:
- Sensor features: speed, steering, acceleration, gyro, radar
- Visual features: lane deviation, vehicle count, lead vehicle distance
- Derived features: speed change, steering jerk, time-to-collision

Output: CSV file with one row per frame, ready for ML training.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.data_loader import Comma2k19Loader
from src.features.synchronizer import DataSynchronizer
from src.image_processing.lane_detector import LaneDetector
from src.image_processing.vehicle_tracker import VehicleTracker


class FeatureExtractor:
    """
    Master Feature Extractor that combines visual and sensor features.
    
    Usage:
        extractor = FeatureExtractor()
        df = extractor.extract_segment_features("data/raw/Chunk_1/route/0")
        df.to_csv("data/processed/features.csv", index=False)
    """
    
    def __init__(self, 
                 use_visual_features: bool = True,
                 skip_frames: int = 1,
                 verbose: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            use_visual_features: Whether to run CV models (slower but more features)
            skip_frames: Process every Nth frame (1 = all frames)
            verbose: Print progress information
        """
        self.use_visual_features = use_visual_features
        self.skip_frames = skip_frames
        self.verbose = verbose
        
        # Initialize CV models if needed
        if use_visual_features:
            if self.verbose:
                print("Initializing visual feature extractors...")
            self.lane_detector = LaneDetector()
            self.vehicle_tracker = VehicleTracker()
        else:
            self.lane_detector = None
            self.vehicle_tracker = None
        
        # Image dimensions (will be set when processing)
        self.image_width = None
        self.image_height = None
        
        # Constants for distance estimation
        self.FOCAL_LENGTH = 910  # Approximate focal length in pixels
        self.CAR_WIDTH = 1.8    # Average car width in meters
    
    def _calculate_lane_deviation(self, frame: np.ndarray) -> float:
        """
        Calculate lateral deviation from lane center.
        
        Args:
            frame: BGR image
            
        Returns:
            Deviation in meters (positive = right, negative = left)
            Returns NaN if lanes cannot be detected
        """
        try:
            # Get the lane-annotated frame and extract line info
            height, width = frame.shape[:2]
            
            # Convert to grayscale and detect edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            
            # ROI mask (lower portion of image)
            mask = np.zeros_like(edges)
            hood_cutoff = 150
            polygon = np.array([[
                (100, height - hood_cutoff),
                (width - 100, height - hood_cutoff),
                (width // 2 + 50, height // 2 + 60),
                (width // 2 - 50, height // 2 + 60)
            ]])
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Hough transform
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30,
                                   minLineLength=20, maxLineGap=20)
            
            if lines is None or len(lines) < 2:
                return np.nan
            
            # Separate left and right lines by slope
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter by slope (ignore horizontal lines)
                if abs(slope) < 0.3:
                    continue
                
                if slope < 0:  # Left lane (negative slope in image coords)
                    left_lines.append(line[0])
                else:  # Right lane
                    right_lines.append(line[0])
            
            if not left_lines or not right_lines:
                return np.nan
            
            # Average the lines
            left_avg = np.mean(left_lines, axis=0).astype(int)
            right_avg = np.mean(right_lines, axis=0).astype(int)
            
            # Calculate lane center at bottom of ROI
            y_eval = height - hood_cutoff - 50  # Slightly above hood cutoff
            
            # Extrapolate left line to y_eval
            if left_avg[3] - left_avg[1] != 0:
                left_slope = (left_avg[2] - left_avg[0]) / (left_avg[3] - left_avg[1])
                left_x = left_avg[0] + left_slope * (y_eval - left_avg[1])
            else:
                left_x = left_avg[0]
            
            # Extrapolate right line to y_eval
            if right_avg[3] - right_avg[1] != 0:
                right_slope = (right_avg[2] - right_avg[0]) / (right_avg[3] - right_avg[1])
                right_x = right_avg[0] + right_slope * (y_eval - right_avg[1])
            else:
                right_x = right_avg[0]
            
            # Lane center and deviation
            lane_center = (left_x + right_x) / 2
            image_center = width / 2
            deviation_pixels = lane_center - image_center
            
            # Convert to meters (approximate)
            # Assume lane width is ~3.7m at the evaluation line
            lane_width_pixels = abs(right_x - left_x)
            if lane_width_pixels > 0:
                meters_per_pixel = 3.7 / lane_width_pixels
                deviation_meters = deviation_pixels * meters_per_pixel
            else:
                deviation_meters = np.nan
            
            return float(deviation_meters)
            
        except Exception as e:
            return np.nan
    
    def _detect_vehicles(self, frame: np.ndarray) -> Tuple[int, float]:
        """
        Detect vehicles and estimate distance to lead vehicle.
        
        Args:
            frame: BGR image
            
        Returns:
            Tuple of (vehicle_count, lead_vehicle_distance)
            lead_vehicle_distance is NaN if no vehicle ahead
        """
        try:
            # Get detections
            results = self.vehicle_tracker.model(
                frame, 
                classes=self.vehicle_tracker.target_classes,
                conf=0.3,
                verbose=False
            )
            
            vehicle_count = 0
            lead_distance = np.nan
            min_y_center = float('inf')  # Track highest vehicle (closest to horizon = furthest)
            
            height, width = frame.shape[:2]
            image_center_x = width / 2
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_count += 1
                    
                    # Check if vehicle is roughly in front (center third of image)
                    box_center_x = (x1 + x2) / 2
                    if abs(box_center_x - image_center_x) < width / 3:
                        # Estimate distance using bounding box width
                        box_width = x2 - x1
                        if box_width > 0:
                            # Distance = (Focal Length * Real Width) / Image Width
                            distance = (self.FOCAL_LENGTH * self.CAR_WIDTH) / box_width
                            
                            # Track the closest vehicle (largest y2 = bottom of bbox)
                            if y2 > min_y_center:
                                min_y_center = y2
                                lead_distance = distance
            
            return vehicle_count, float(lead_distance)
            
        except Exception as e:
            return 0, np.nan
    
    def extract_frame_features(self, 
                               frame: np.ndarray, 
                               frame_data: 'SynchronizedFrame') -> Dict:
        """
        Extract all features for a single frame.
        
        Args:
            frame: BGR image
            frame_data: Synchronized sensor data for this frame
            
        Returns:
            Dict of feature name -> value
        """
        features = {
            # Frame metadata
            'frame_idx': frame_data.frame_idx,
            'timestamp': frame_data.timestamp,
            
            # Sensor features
            'speed': frame_data.speed,
            'steering': frame_data.steering,
            'accel_forward': frame_data.acceleration[0] if len(frame_data.acceleration) > 0 else 0,
            'accel_lateral': frame_data.acceleration[1] if len(frame_data.acceleration) > 1 else 0,
            'accel_vertical': frame_data.acceleration[2] if len(frame_data.acceleration) > 2 else 0,
            'gyro_yaw': frame_data.gyro[2] if len(frame_data.gyro) > 2 else 0,
            'radar_distance': frame_data.radar_distance,
            'radar_rel_speed': frame_data.radar_speed,
        }
        
        # Visual features (optional, computationally expensive)
        if self.use_visual_features and frame is not None:
            # Lane deviation
            features['lane_deviation'] = self._calculate_lane_deviation(frame)
            
            # Vehicle detection
            vehicle_count, lead_distance = self._detect_vehicles(frame)
            features['vehicle_count'] = vehicle_count
            features['lead_distance_visual'] = lead_distance
        else:
            features['lane_deviation'] = np.nan
            features['vehicle_count'] = 0
            features['lead_distance_visual'] = np.nan
        
        return features
    
    def extract_segment_features(self, 
                                  segment_path: str,
                                  max_frames: int = None) -> pd.DataFrame:
        """
        Extract features from an entire segment.
        
        Args:
            segment_path: Path to segment directory
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            DataFrame with one row per processed frame
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing segment: {segment_path}")
            print(f"{'='*60}")
        
        # Load data
        loader = Comma2k19Loader(segment_path)
        
        if self.verbose:
            print("\nLoading telemetry...")
        telemetry = loader.load_telemetry()
        frame_times = loader.load_frame_times()
        
        # Create synchronizer
        sync = DataSynchronizer(frame_times, telemetry)
        
        if self.verbose:
            print(f"Segment duration: {sync.get_duration():.1f}s")
            print(f"Total frames: {sync.num_frames}")
        
        # Compute derived features
        derived = sync.compute_derived_features()
        
        # Process frames
        features_list = []
        
        if self.use_visual_features:
            # Process with video
            if self.verbose:
                print("\nProcessing frames with visual features...")
            
            frame_iter = loader.iter_frames()
            if max_frames:
                total = min(max_frames, sync.num_frames)
            else:
                total = sync.num_frames
            
            pbar = tqdm(total=total // self.skip_frames, disable=not self.verbose)
            
            for frame, frame_idx in frame_iter:
                if max_frames and frame_idx >= max_frames:
                    break
                
                if frame_idx % self.skip_frames != 0:
                    continue
                
                frame_data = sync.get_frame_data(frame_idx)
                features = self.extract_frame_features(frame, frame_data)
                
                # Add derived features
                if 'speed_change' in derived:
                    features['speed_change'] = derived['speed_change'][frame_idx]
                if 'steering_rate' in derived:
                    features['steering_rate'] = derived['steering_rate'][frame_idx]
                if 'steering_jerk' in derived:
                    features['steering_jerk'] = derived['steering_jerk'][frame_idx]
                
                features_list.append(features)
                pbar.update(1)
            
            pbar.close()
        else:
            # Process without video (sensor data only)
            if self.verbose:
                print("\nProcessing sensor data only (no visual features)...")
            
            indices = range(0, sync.num_frames, self.skip_frames)
            if max_frames:
                indices = range(0, min(max_frames, sync.num_frames), self.skip_frames)
            
            for frame_idx in tqdm(indices, disable=not self.verbose):
                frame_data = sync.get_frame_data(frame_idx)
                features = self.extract_frame_features(None, frame_data)
                
                # Add derived features
                if 'speed_change' in derived:
                    features['speed_change'] = derived['speed_change'][frame_idx]
                if 'steering_rate' in derived:
                    features['steering_rate'] = derived['steering_rate'][frame_idx]
                if 'steering_jerk' in derived:
                    features['steering_jerk'] = derived['steering_jerk'][frame_idx]
                
                features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        if self.verbose:
            print(f"\nExtracted {len(df)} feature rows")
            print(f"Columns: {list(df.columns)}")
        
        return df
    
    def extract_multiple_segments(self,
                                   segment_paths: List[str],
                                   output_path: str = None) -> pd.DataFrame:
        """
        Extract features from multiple segments.
        
        Args:
            segment_paths: List of segment directory paths
            output_path: Optional path to save combined CSV
            
        Returns:
            Combined DataFrame from all segments
        """
        all_dfs = []
        
        for i, segment_path in enumerate(segment_paths):
            if self.verbose:
                print(f"\n[{i+1}/{len(segment_paths)}] Processing {segment_path}")
            
            try:
                df = self.extract_segment_features(segment_path)
                df['segment_id'] = i
                df['segment_path'] = segment_path
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {segment_path}: {e}")
                continue
        
        if not all_dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if output_path:
            combined_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"\nSaved combined features to {output_path}")
        
        return combined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from Comma2k19 segments")
    parser.add_argument("--segment", type=str, required=True, help="Path to segment")
    parser.add_argument("--output", type=str, default="data/processed/features.csv", 
                       help="Output CSV path")
    parser.add_argument("--no-visual", action="store_true", 
                       help="Skip visual features (faster)")
    parser.add_argument("--skip-frames", type=int, default=1,
                       help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process")
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor(
        use_visual_features=not args.no_visual,
        skip_frames=args.skip_frames
    )
    
    df = extractor.extract_segment_features(args.segment, max_frames=args.max_frames)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Features saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
