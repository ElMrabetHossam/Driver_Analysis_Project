"""
Data Synchronizer

Synchronizes video frames with sensor telemetry data using timestamp interpolation.

The Comma2k19 dataset logs sensors at variable rates:
- Speed/Steering: ~100 Hz
- IMU: ~100 Hz  
- Radar: ~20 Hz
- Video: ~20 FPS

This module aligns all data to video frame timestamps.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SynchronizedFrame:
    """Container for synchronized frame data."""
    frame_idx: int
    timestamp: float
    speed: float               # m/s
    steering: float            # degrees
    acceleration: np.ndarray   # [forward, right, down] m/s^2
    gyro: np.ndarray           # [forward, right, down] rad/s
    radar_distance: float      # distance to lead vehicle (m), NaN if no vehicle
    radar_speed: float         # relative speed to lead vehicle (m/s)


class DataSynchronizer:
    """
    Synchronize sensor data to video frame timestamps.
    
    Usage:
        sync = DataSynchronizer(frame_times, telemetry)
        for frame_idx in range(num_frames):
            data = sync.get_frame_data(frame_idx)
            print(f"Frame {data.frame_idx}: speed={data.speed:.1f} m/s")
    """
    
    def __init__(self, frame_times: np.ndarray, telemetry: Dict):
        """
        Initialize synchronizer.
        
        Args:
            frame_times: Array of frame timestamps (from frame_times.npy)
            telemetry: Dict of TelemetryData from Comma2k19Loader.load_telemetry()
        """
        self.frame_times = frame_times
        self.telemetry = telemetry
        self.num_frames = len(frame_times)
        
        # Pre-compute synchronized arrays for efficiency
        self._sync_cache = {}
    
    def _interpolate_1d(self, sensor_times: np.ndarray, sensor_values: np.ndarray) -> np.ndarray:
        """Interpolate 1D sensor values to frame times."""
        return np.interp(self.frame_times, sensor_times, sensor_values)
    
    def _interpolate_nd(self, sensor_times: np.ndarray, sensor_values: np.ndarray) -> np.ndarray:
        """Interpolate multi-dimensional sensor values to frame times."""
        if sensor_values.ndim == 1:
            return self._interpolate_1d(sensor_times, sensor_values)
        
        result = np.zeros((self.num_frames, sensor_values.shape[1]))
        for i in range(sensor_values.shape[1]):
            result[:, i] = np.interp(self.frame_times, sensor_times, sensor_values[:, i])
        return result
    
    def sync_sensor(self, sensor_name: str) -> Optional[np.ndarray]:
        """
        Get synchronized sensor data aligned to frame times.
        
        Args:
            sensor_name: Name of the sensor (e.g., 'speed', 'steering')
            
        Returns:
            Array of shape (num_frames,) or (num_frames, dims) with interpolated values
        """
        if sensor_name in self._sync_cache:
            return self._sync_cache[sensor_name]
        
        if sensor_name not in self.telemetry:
            return None
        
        data = self.telemetry[sensor_name]
        synced = self._interpolate_nd(data.timestamps, data.values)
        
        self._sync_cache[sensor_name] = synced
        return synced
    
    def sync_all(self) -> Dict[str, np.ndarray]:
        """
        Synchronize all available sensors.
        
        Returns:
            Dict mapping sensor names to synchronized arrays
        """
        synced = {}
        for name in self.telemetry.keys():
            result = self.sync_sensor(name)
            if result is not None:
                synced[name] = result
        return synced
    
    def get_frame_data(self, frame_idx: int) -> SynchronizedFrame:
        """
        Get all synchronized sensor data for a specific frame.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            SynchronizedFrame with all sensor values
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")
        
        timestamp = self.frame_times[frame_idx]
        
        # Get speed
        speed_arr = self.sync_sensor('speed')
        speed = float(speed_arr[frame_idx]) if speed_arr is not None else 0.0
        
        # Get steering
        steering_arr = self.sync_sensor('steering')
        steering = float(steering_arr[frame_idx]) if steering_arr is not None else 0.0
        
        # Get acceleration
        accel_arr = self.sync_sensor('acceleration')
        acceleration = accel_arr[frame_idx] if accel_arr is not None else np.zeros(3)
        
        # Get gyro
        gyro_arr = self.sync_sensor('gyro')
        gyro = gyro_arr[frame_idx] if gyro_arr is not None else np.zeros(3)
        
        # Get radar (more complex - find closest detection)
        radar_distance = np.nan
        radar_speed = np.nan
        
        if 'radar' in self.telemetry:
            radar_data = self.telemetry['radar']
            # Find radar readings closest to this frame's timestamp
            time_diff = np.abs(radar_data.timestamps - timestamp)
            closest_idx = np.argmin(time_diff)
            
            if time_diff[closest_idx] < 0.1:  # Within 100ms
                radar_vals = radar_data.values[closest_idx]
                # Radar format: [forward_dist, left_dist, rel_speed, nan, nan, address, new_track]
                if len(radar_vals) >= 3:
                    radar_distance = float(radar_vals[0])  # Forward distance
                    radar_speed = float(radar_vals[2])     # Relative speed
        
        return SynchronizedFrame(
            frame_idx=frame_idx,
            timestamp=float(timestamp),
            speed=speed,
            steering=steering,
            acceleration=acceleration,
            gyro=gyro,
            radar_distance=radar_distance,
            radar_speed=radar_speed
        )
    
    def compute_derived_features(self) -> Dict[str, np.ndarray]:
        """
        Compute derived features from raw sensor data.
        
        Returns:
            Dict with derived features:
            - speed_change: rate of speed change (m/s^2)
            - steering_rate: rate of steering change (deg/s)
            - jerk: rate of acceleration change (m/s^3)
        """
        derived = {}
        
        # Compute time deltas
        dt = np.diff(self.frame_times)
        dt = np.append(dt, dt[-1])  # Pad to match frame count
        dt = np.maximum(dt, 0.001)  # Avoid division by zero
        
        # Speed change (acceleration from speed)
        speed = self.sync_sensor('speed')
        if speed is not None:
            # Ensure 1D array
            if speed.ndim > 1:
                speed = speed.flatten() if speed.shape[1] == 1 else speed[:, 0]
            speed_change = np.diff(speed, prepend=speed[0]) / dt
            derived['speed_change'] = speed_change
        
        # Steering rate
        steering = self.sync_sensor('steering')
        if steering is not None:
            # Ensure 1D array
            if steering.ndim > 1:
                steering = steering.flatten() if steering.shape[1] == 1 else steering[:, 0]
            steering_rate = np.diff(steering, prepend=steering[0]) / dt
            derived['steering_rate'] = steering_rate
            
            # Steering jerk (rate of rate change)
            steering_jerk = np.diff(steering_rate, prepend=steering_rate[0]) / dt
            derived['steering_jerk'] = steering_jerk
        
        # Jerk from acceleration
        accel = self.sync_sensor('acceleration')
        if accel is not None and accel.ndim > 1:
            jerk = np.diff(accel, axis=0, prepend=accel[0:1]) / dt[:, np.newaxis]
            derived['jerk'] = jerk
        
        return derived
    
    def get_time_range(self) -> tuple:
        """Get the time range covered by this segment."""
        return float(self.frame_times[0]), float(self.frame_times[-1])
    
    def get_duration(self) -> float:
        """Get segment duration in seconds."""
        return float(self.frame_times[-1] - self.frame_times[0])


def create_synchronized_dataset(loader, sync: DataSynchronizer) -> Dict[str, np.ndarray]:
    """
    Create a complete synchronized dataset from a segment.
    
    Args:
        loader: Comma2k19Loader instance
        sync: DataSynchronizer instance
        
    Returns:
        Dict with all synchronized data ready for feature engineering
    """
    dataset = {
        'frame_times': sync.frame_times,
        'num_frames': sync.num_frames,
        'duration': sync.get_duration(),
    }
    
    # Add synchronized sensor data
    synced = sync.sync_all()
    for name, values in synced.items():
        dataset[name] = values
    
    # Add derived features
    derived = sync.compute_derived_features()
    for name, values in derived.items():
        dataset[name] = values
    
    return dataset


if __name__ == "__main__":
    from data_loader import Comma2k19Loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data synchronization")
    parser.add_argument("--segment", type=str, required=True, help="Path to segment")
    
    args = parser.parse_args()
    
    print("Loading segment...")
    loader = Comma2k19Loader(args.segment)
    
    frame_times = loader.load_frame_times()
    telemetry = loader.load_telemetry()
    
    print(f"\nFrame times: {len(frame_times)} frames")
    print(f"Duration: {frame_times[-1] - frame_times[0]:.1f} seconds")
    
    print("\nSynchronizing...")
    sync = DataSynchronizer(frame_times, telemetry)
    
    # Test getting frame data
    print("\n=== Sample Frame Data ===")
    for i in [0, len(frame_times)//2, len(frame_times)-1]:
        data = sync.get_frame_data(i)
        print(f"Frame {data.frame_idx}: t={data.timestamp:.2f}s, "
              f"speed={data.speed:.1f}m/s, steering={data.steering:.1f}deg")
    
    # Test derived features
    print("\n=== Derived Features ===")
    derived = sync.compute_derived_features()
    for name, values in derived.items():
        if values.ndim == 1:
            print(f"{name}: mean={values.mean():.3f}, std={values.std():.3f}")
        else:
            print(f"{name}: shape={values.shape}")
