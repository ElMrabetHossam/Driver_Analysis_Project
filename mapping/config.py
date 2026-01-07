"""
Configuration file for Vehicle Tracking Visualization System
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw', 'comma2k19', 'scb4', 'global_pose')

# Data files
DATA_FILES = {
    'positions': os.path.join(DATA_DIR, 'frame_positions'),
    'velocities': os.path.join(DATA_DIR, 'frame_velocities'),
    'orientations': os.path.join(DATA_DIR, 'frame_orientations'),
    'times': os.path.join(DATA_DIR, 'frame_times'),
    'gps_times': os.path.join(DATA_DIR, 'frame_gps_times')
}

# Video and Model paths
VIDEO_FILE = os.path.join(BASE_DIR, '..', 'data', 'raw', 'comma2k19', 'scb4', 'video.mp4')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'yolov8n.pt')


# Visualization settings
MAP_SETTINGS = {
    'default_zoom': 15,
    'tile_style': 'open-street-map',  # changed to Plotly-supported style name
    'trajectory_color': '#2E86DE',
    'trajectory_weight': 3,
    'start_marker_color': '#10AC84',
    'end_marker_color': '#EE5A6F',
    'vehicle_marker_color': '#FF6B6B'
}

# Camera calibration defaults (used to project ground points into image)
# If you have real calibration files, replace these with loaded matrices.
CAMERA = {
    # focal length in pixels (approx) and principal point
    'focal_length_px': 910,
    'principal_point': None,  # (cx, cy) -> if None will use image center at runtime
    'camera_height_m': 1.2,
    # Optional extrinsic: camera->vehicle transform (3x4 or 4x4). If None, assume camera at origin.
    'extrinsic_matrix': None,
    # Optional intrinsic matrix K as 3x3 numpy array saved on disk. Provide path if available.
    'intrinsic_path': ''
}

# Radar & wheel settings
RADAR_SETTINGS = {
    'max_targets': 32,
    'display_scale': 1.0
}

WHEEL_SETTINGS = {
    'wheelbase_m': 2.7,
    'track_width_m': 1.6
}

# Mapbox token (optional). Read from environment if set.
MAPBOX_TOKEN = os.environ.get('MAPBOX_TOKEN', '')

# Animation settings
ANIMATION_SETTINGS = {
    'default_speed': 1.0,  # 1.0 = real-time
    'min_speed': 0.1,
    'max_speed': 10.0,
    'fps': 30,  # Frames per second for smooth animation
    'interpolation': True  # Smooth interpolation between data points
}

# Dashboard settings
DASHBOARD_SETTINGS = {
    'title': 'Vehicle Tracking System',
    'update_interval_ms': 50,  # Milliseconds between dashboard updates
    'theme': 'dark',  # 'dark' or 'light'
    'show_telemetry_graphs': True
}

# Metric display settings
METRIC_UNITS = {
    'speed': 'km/h',
    'altitude': 'm',
    'distance': 'km',
    'time': 's'
}

# Colors for dashboard (dark theme)
COLORS = {
    'background': '#1E1E1E',
    'card_background': '#2D2D2D',
    'text': '#FFFFFF',
    'text_secondary': '#B0B0B0',
    'primary': '#4A90E2',
    'secondary': '#00d2ff',  # Cyan for secondary elements
    'success': '#10AC84',
    'warning': '#F2C23C',
    'danger': '#EE5A6F',
    'chart_line': '#4A90E2',
    'chart_grid': '#404040'
}
