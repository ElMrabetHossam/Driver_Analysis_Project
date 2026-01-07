"""
Metrics Calculator Module
Calculates speed, distance, and other telemetry metrics from vehicle data
"""
import numpy as np


class MetricsCalculator:
    """Calculates various metrics from vehicle tracking data"""
    
    def __init__(self, velocities, times, gps_coords):
        """
        Initialize metrics calculator
        
        Args:
            velocities: numpy array of velocity vectors (n, 3)
            times: numpy array of timestamps
            gps_coords: numpy array of GPS coordinates (n, 3) - lat, lon, alt
        """
        self.velocities = velocities
        self.times = times
        self.gps_coords = gps_coords
        self.num_frames = len(times)
    
    def calculate_speed_from_velocity(self):
        """
        Calculate speed magnitude from velocity vectors
        
        Returns:
            numpy array of speeds in m/s
        """
        # Speed is magnitude of velocity vector
        speeds = np.linalg.norm(self.velocities, axis=1)
        return speeds
    
    def speed_ms_to_kmh(self, speed_ms):
        """Convert speed from m/s to km/h"""
        return speed_ms * 3.6
    
    def speed_ms_to_mph(self, speed_ms):
        """Convert speed from m/s to mph"""
        return speed_ms * 2.23694
    
    def get_speeds_kmh(self):
        """Get speeds in km/h"""
        speeds_ms = self.calculate_speed_from_velocity()
        return self.speed_ms_to_kmh(speeds_ms)
    
    def calculate_acceleration(self):
        """
        Calculate acceleration from velocity changes
        
        Returns:
            numpy array of accelerations in m/s²
        """
        accelerations = np.zeros(self.num_frames)
        
        for i in range(1, self.num_frames):
            dv = self.velocities[i] - self.velocities[i - 1]
            dt = self.times[i] - self.times[i - 1]
            
            if dt > 0:
                # Magnitude of acceleration
                accelerations[i] = np.linalg.norm(dv) / dt
        
        return accelerations
    
    def calculate_time_deltas(self):
        """
        Calculate time differences between frames
        
        Returns:
            numpy array of time deltas in seconds
        """
        time_deltas = np.zeros(self.num_frames)
        time_deltas[1:] = np.diff(self.times)
        return time_deltas
    
    def calculate_altitude_changes(self):
        """
        Calculate altitude changes
        
        Returns:
            numpy array of altitude changes in meters
        """
        altitudes = self.gps_coords[:, 2]
        altitude_changes = np.zeros(self.num_frames)
        altitude_changes[1:] = np.diff(altitudes)
        return altitude_changes
    
    def get_statistics(self):
        """
        Get comprehensive statistics about the journey
        
        Returns:
            dict with various statistics
        """
        speeds_kmh = self.get_speeds_kmh()
        accelerations = self.calculate_acceleration()
        time_deltas = self.calculate_time_deltas()
        altitudes = self.gps_coords[:, 2]
        
        stats = {
            'duration_seconds': self.times[-1] - self.times[0],
            'speed': {
                'max_kmh': np.max(speeds_kmh),
                'min_kmh': np.min(speeds_kmh),
                'avg_kmh': np.mean(speeds_kmh),
                'median_kmh': np.median(speeds_kmh)
            },
            'acceleration': {
                'max_ms2': np.max(accelerations),
                'avg_ms2': np.mean(accelerations)
            },
            'altitude': {
                'max_m': np.max(altitudes),
                'min_m': np.min(altitudes),
                'start_m': altitudes[0],
                'end_m': altitudes[-1],
                'change_m': altitudes[-1] - altitudes[0]
            },
            'timing': {
                'avg_frame_interval_ms': np.mean(time_deltas[1:]) * 1000,
                'min_frame_interval_ms': np.min(time_deltas[1:]) * 1000,
                'max_frame_interval_ms': np.max(time_deltas[1:]) * 1000
            }
        }
        
        return stats
    
    def get_frame_metrics(self, frame_idx):
        """
        Get all metrics for a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            dict with frame metrics
        """
        speeds_ms = self.calculate_speed_from_velocity()
        speeds_kmh = self.get_speeds_kmh()
        accelerations = self.calculate_acceleration()
        time_deltas = self.calculate_time_deltas()
        
        metrics = {
            'time': self.times[frame_idx],
            'position': {
                'lat': self.gps_coords[frame_idx, 0],
                'lon': self.gps_coords[frame_idx, 1],
                'alt': self.gps_coords[frame_idx, 2]
            },
            'velocity': {
                'vx': self.velocities[frame_idx, 0],
                'vy': self.velocities[frame_idx, 1],
                'vz': self.velocities[frame_idx, 2],
                'magnitude_ms': speeds_ms[frame_idx],
                'magnitude_kmh': speeds_kmh[frame_idx]
            },
            'acceleration_ms2': accelerations[frame_idx],
            'time_delta': time_deltas[frame_idx]
        }
        
        return metrics


if __name__ == "__main__":
    # Test with dummy data
    velocities = np.random.rand(100, 3) * 10
    times = np.linspace(0, 100, 100)
    gps_coords = np.random.rand(100, 3)
    gps_coords[:, 2] *= 100  # Altitudes
    
    calc = MetricsCalculator(velocities, times, gps_coords)
    
    print("=== Statistics ===")
    stats = calc.get_statistics()
    print(f"Duration: {stats['duration_seconds']:.2f}s")
    print(f"Max speed: {stats['speed']['max_kmh']:.2f} km/h")
    print(f"Avg speed: {stats['speed']['avg_kmh']:.2f} km/h")
