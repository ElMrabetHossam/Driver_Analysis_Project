"""
Comma2k19 Data Loader

Loads video frames and telemetry data from Comma2k19 dataset segments.

The dataset structure is:
    segment/
    ├── video.hevc              # Video file (~1 minute)
    ├── processed_log/          # Sensor data as numpy arrays
    │   ├── CAN/
    │   │   ├── car_speed/      # [t.npy, value.npy]
    │   │   ├── steering_angle/
    │   │   ├── wheel_speeds/
    │   │   └── radar/
    │   └── IMU/
    │       ├── acceleration/
    │       ├── gyro/
    │       └── gyro_uncalibrated/
    └── global_pos/             # Camera pose data
        ├── frame_times.npy     # Timestamps for each video frame
        ├── frame_positions.npy
        └── frame_orientations.npy
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Generator, Optional
from dataclasses import dataclass


@dataclass
class TelemetryData:
    """Container for sensor telemetry data."""
    timestamps: np.ndarray  # Time in seconds (boot time)
    values: np.ndarray      # Sensor values
    
    def __len__(self):
        return len(self.timestamps)
    
    def get_at_time(self, t: float) -> np.ndarray:
        """Get interpolated value at a specific time."""
        if self.values.ndim == 1:
            return np.interp(t, self.timestamps, self.values)
        else:
            # Multi-dimensional: interpolate each column
            result = np.zeros(self.values.shape[1])
            for i in range(self.values.shape[1]):
                result[i] = np.interp(t, self.timestamps, self.values[:, i])
            return result


class Comma2k19Loader:
    """
    Load video frames and telemetry from a Comma2k19 segment.
    
    Usage:
        loader = Comma2k19Loader("data/raw/Chunk_1/route_id/0")
        telemetry = loader.load_telemetry()
        
        for frame, timestamp in loader.iter_frames():
            speed = telemetry['speed'].get_at_time(timestamp)
            # Process frame with synchronized speed
    """
    
    # Standard sensors available in the dataset
    # Note: Some naming variations exist between sample and full dataset
    SENSOR_PATHS = {
        'speed': ['CAN/speed', 'CAN/car_speed'],  # Alternative names
        'steering': ['CAN/steering_angle'], 
        'wheel_speeds': ['CAN/wheel_speed', 'CAN/wheel_speeds'],
        'radar': ['CAN/radar'],
        'acceleration': ['IMU/accelerometer', 'IMU/acceleration'],
        'gyro': ['IMU/gyro'],
        'gyro_uncalibrated': ['IMU/gyro_uncalibrated'],
        'gyro_bias': ['IMU/gyro_bias'],
        'magnetic': ['IMU/magnetometer', 'IMU/magnetic'],
    }
    
    def __init__(self, segment_path: str):
        """
        Initialize the loader.
        
        Args:
            segment_path: Path to a segment directory (e.g., "data/raw/Chunk_1/route_id/0")
        """
        self.segment_path = Path(segment_path)
        self.processed_log = self.segment_path / "processed_log"
        
        # Handle both naming conventions: global_pos and global_pose
        if (self.segment_path / "global_pos").exists():
            self.global_pos = self.segment_path / "global_pos"
        else:
            self.global_pos = self.segment_path / "global_pose"
        
        self.video_path = self.segment_path / "video.hevc"
        
        if not self.segment_path.exists():
            raise FileNotFoundError(f"Segment not found: {self.segment_path}")
    
    def _load_sensor(self, sensor_subpaths: list) -> Optional[TelemetryData]:
        """
        Load a sensor's timestamp and value arrays.
        
        Args:
            sensor_subpaths: List of alternative paths within processed_log
            
        Returns:
            TelemetryData object or None if sensor not available
        """
        # Try each alternative path
        sensor_dir = None
        for subpath in sensor_subpaths:
            candidate = self.processed_log / subpath
            if candidate.exists():
                sensor_dir = candidate
                break
        
        if sensor_dir is None:
            return None
        
        # Look for t and value files (various naming conventions)
        t_file = None
        value_file = None
        
        # Try standard naming
        for t_name in ['t', 't.npy']:
            candidate = sensor_dir / t_name
            if candidate.exists():
                t_file = candidate
                break
        
        for v_name in ['value', 'value.npy']:
            candidate = sensor_dir / v_name
            if candidate.exists():
                value_file = candidate
                break
        
        # If not found, try to find any files
        if t_file is None or value_file is None:
            files = sorted([f for f in sensor_dir.iterdir() if f.is_file()])
            if len(files) >= 2:
                # Assume first alphabetically is timestamps, second is values
                t_file = files[0]
                value_file = files[1]
            elif len(files) == 1:
                # Single file might contain both
                return None
            else:
                return None
        
        try:
            timestamps = np.load(t_file)
            values = np.load(value_file)
            return TelemetryData(timestamps=timestamps, values=values)
        except Exception as e:
            print(f"Warning: Failed to load {sensor_dir}: {e}")
            return None
    
    def load_telemetry(self) -> Dict[str, TelemetryData]:
        """
        Load all available telemetry data.
        
        Returns:
            Dictionary mapping sensor names to TelemetryData objects
        """
        telemetry = {}
        
        for name, path in self.SENSOR_PATHS.items():
            data = self._load_sensor(path)
            if data is not None:
                telemetry[name] = data
                print(f"  Loaded {name}: {len(data)} samples")
        
        return telemetry
    
    def load_frame_times(self) -> np.ndarray:
        """
        Load timestamps for each video frame.
        
        Returns:
            Array of frame timestamps in seconds (boot time)
        """
        # Try different file names/extensions
        for name in ['frame_times.npy', 'frame_times']:
            frame_times_path = self.global_pos / name
            if frame_times_path.exists():
                return np.load(frame_times_path)
        
        raise FileNotFoundError(f"Frame times not found in: {self.global_pos}")
    
    def load_frame_positions(self) -> np.ndarray:
        """
        Load global positions (ECEF) for each frame.
        
        Returns:
            Array of shape (num_frames, 3) with [x, y, z] positions
        """
        positions_path = self.global_pos / "frame_positions.npy"
        
        if positions_path.exists():
            return np.load(positions_path)
        return None
    
    def get_video_info(self) -> Dict:
        """
        Get video metadata.
        
        Returns:
            Dict with fps, width, height, frame_count
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        
        cap.release()
        return info
    
    def iter_frames(self, start_frame: int = 0, end_frame: int = None) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Iterate over video frames.
        
        Args:
            start_frame: First frame to read (0-indexed)
            end_frame: Last frame to read (exclusive), None for all frames
            
        Yields:
            Tuple of (frame, frame_index)
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        
        while cap.isOpened():
            if end_frame is not None and frame_idx >= end_frame:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            yield frame, frame_idx
            frame_idx += 1
        
        cap.release()
    
    def iter_frames_with_time(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Iterate over video frames with their timestamps.
        
        Yields:
            Tuple of (frame, frame_index, timestamp)
        """
        frame_times = self.load_frame_times()
        
        for frame, idx in self.iter_frames():
            if idx < len(frame_times):
                yield frame, idx, frame_times[idx]
            else:
                break
    
    def get_segment_info(self) -> Dict:
        """
        Get comprehensive segment information.
        
        Returns:
            Dict with all segment metadata
        """
        info = {
            'path': str(self.segment_path),
            'video_exists': self.video_path.exists(),
            'processed_log_exists': self.processed_log.exists(),
            'global_pos_exists': self.global_pos.exists(),
        }
        
        if self.video_path.exists():
            info['video'] = self.get_video_info()
        
        if self.processed_log.exists():
            info['available_sensors'] = []
            for name, paths in self.SENSOR_PATHS.items():
                # Check any of the alternative paths
                for path in paths:
                    if (self.processed_log / path).exists():
                        info['available_sensors'].append(name)
                        break
        
        if self.global_pos.exists():
            try:
                frame_times = self.load_frame_times()
                info['duration_seconds'] = float(frame_times[-1] - frame_times[0])
            except:
                pass
        
        return info


def find_all_segments(data_dir: str) -> List[Path]:
    """
    Find all segments in a data directory.
    
    Args:
        data_dir: Path to data/raw directory
        
    Returns:
        List of segment paths
    """
    data_path = Path(data_dir)
    segments = []
    
    # Look for Chunk_N directories
    for chunk in data_path.glob("Chunk_*"):
        if chunk.is_dir():
            # Look for route directories (format: dongle_id|timestamp)
            for route in chunk.iterdir():
                if route.is_dir() and '|' in route.name:
                    # Look for segment directories (numbered)
                    for segment in sorted(route.iterdir()):
                        if segment.is_dir() and segment.name.isdigit():
                            segments.append(segment)
    
    return segments


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comma2k19 Data Loader")
    parser.add_argument("--segment", type=str, help="Path to a specific segment")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--list", action="store_true", help="List all available segments")
    
    args = parser.parse_args()
    
    if args.list:
        segments = find_all_segments(args.data_dir)
        print(f"Found {len(segments)} segments:")
        for seg in segments[:10]:
            print(f"  {seg}")
        if len(segments) > 10:
            print(f"  ... and {len(segments) - 10} more")
    
    elif args.segment:
        loader = Comma2k19Loader(args.segment)
        info = loader.get_segment_info()
        
        print("\n=== Segment Info ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print("\n=== Loading Telemetry ===")
        telemetry = loader.load_telemetry()
        
        print("\n=== Sample Data ===")
        if 'speed' in telemetry:
            speed = telemetry['speed']
            print(f"Speed: min={speed.values.min():.1f}, max={speed.values.max():.1f} m/s")
        
        if 'steering' in telemetry:
            steering = telemetry['steering']
            print(f"Steering: min={steering.values.min():.1f}, max={steering.values.max():.1f} deg")
    
    else:
        print("Use --segment <path> to load a segment or --list to find segments")
