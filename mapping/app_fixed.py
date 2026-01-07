
"""
Main Application Module - Professional Versatile Edition
Vehicle Tracking Visualization System with YOLOv8 & Real-time Telemetry
"""
import os
import sys
import dash
from dash import html, dcc, Input, Output, State, no_update
import numpy as np
import cv2
import base64
from data_loader import VehicleDataLoader
from coordinate_converter import CoordinateConverter
from metrics_calculator import MetricsCalculator
from dynamic_map_generator import DynamicMapGenerator
from video_processor import VideoProcessor
from dashboard_components import (
    create_speed_gauge, create_g_force_meter, create_steering_gauge,
    create_speed_timeline_graph
)
from advanced_dashboard_components import (
    create_attitude_indicator, create_radar_display, create_wheel_dynamics_display,
    create_dynamics_metrics_gauge, create_centrifugal_force_gauge, create_ground_truth_path_display
)
from config import DASHBOARD_SETTINGS, BASE_DIR

# ----------------------------------------------------------------------------
# APP INITIALIZATION
# ----------------------------------------------------------------------------
app = dash.Dash(__name__, update_title=None)
app.title = "Vehicle Mission Control"
server = app.server

class AppData:
    loader = None
    converter = None
    calculator = None
    map_gen = None
    video_proc = None
    gps_coords = None
    headings = None
    speeds = None
    times = None
    is_ready = False

app_data = AppData()

def load_and_process_data():
    if app_data.is_ready: return

    # 1. Load Telemetry Data
    app_data.loader = VehicleDataLoader()
    app_data.loader.load_all_data()
    app_data.loader.load_additional_data()

    # 2. Process Coordinates
    app_data.converter = CoordinateConverter()
    app_data.gps_coords = app_data.converter.ecef_array_to_gps(app_data.loader.positions)
    app_data.headings = app_data.converter.calculate_headings_from_positions(app_data.gps_coords)

    # 3. Calculate Metrics
    app_data.calculator = MetricsCalculator(app_data.loader.velocities, app_data.loader.times, app_data.gps_coords)
    app_data.speeds = app_data.calculator.get_speeds_kmh()
    app_data.times = app_data.loader.times

    # 4. Initialize Map
    app_data.map_gen = DynamicMapGenerator(app_data.gps_coords, app_data.headings)

    # 5. Initialize Video Processor (YOLO)
    # Paths based on user structure
    project_root = os.path.abspath(os.path.join(BASE_DIR, '..'))

    # Locate Video
    # We look for scb4 directory deep in data/raw... or rely on fixed path
    # Using the path found in exploration
    video_path = os.path.join(project_root, 'data', 'raw', 'comma2k19', 'scb4', 'video.mp4')
    if not os.path.exists(video_path):
        # Fallback search if path structure varies
        print("Warning: Direct path to generic video.mp4 failed. Searching...")
        for root, dirs, files in os.walk(project_root):
            if 'video.mp4' in files and 'scb4' in root:
                video_path = os.path.join(root, 'video.mp4')
                break

    # Locate Model
    model_path = os.path.join(project_root, 'yolov8n.pt')

    print(f"Loading Video: {video_path}")
    print(f"Loading Model: {model_path}")

    if os.path.exists(video_path) and os.path.exists(model_path):
        app_data.video_proc = VideoProcessor(video_path, model_path)
    else:
        print(f"CRITICAL: Video or Model not found. V: {os.path.exists(video_path)} M: {os.path.exists(model_path)}")

    app_data.is_ready = True

# ----------------------------------------------------------------------------
# LAYOUT (Clean Grid System)
# ----------------------------------------------------------------------------
def create_layout():
    return html.Div([
        # Initialize playing to True for Auto-Play
        dcc.Store(id='animation-state', data={'frame': 0, 'playing': True}),
        dcc.Interval(id='animation-interval', interval=100, disabled=False),

        # Header
        html.Div([
            html.Div("VEHICLE OS | ULTRA-TECHNICAL DASHBOARD", className='logo-text', style={'fontSize': '20px', 'fontWeight': '900', 'color': '#00d2ff'}),
            html.Div(id='time-display', style={'fontFamily': 'monospace', 'color': '#888'})
        ], className='top-header'),

        # 3x3 Responsive Grid
        html.Div([
            # COL 1, ROW 1: MAP
            html.Div([
                html.Div("SATELLITE TRAJECTORY", className='section-label'),
                dcc.Graph(id='map-display', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

            # COL 2, ROW 1: VIDEO
            html.Div([
                html.Div("OPTICAL FEED (YOLOv8 DETECT)", className='section-label'),
                html.Div([
                    html.Img(id='video-frame-display', className='video-frame')
                ], className='video-container')
            ], className='grid-item'),

            # COL 3, ROW 1: GROUND TRUTH PATH
            html.Div([
                html.Div("GROUND TRUTH PATH", className='section-label'),
                dcc.Graph(id='ground-truth-path-display', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

            # COL 1, ROW 2: SPEED
            html.Div([
                html.Div([
                    html.Span("VELOCITY GAUGE", className='section-label'),
                    html.Span(id='vehicle-count-display', style={'float': 'right', 'color': '#00d2ff', 'fontSize': '12px'})
                ]),
                dcc.Graph(id='speed-gauge', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

            # COL 2, ROW 2: ATTITUDE
            html.Div([
                html.Div("ATTITUDE INDICATOR", className='section-label'),
                dcc.Graph(id='attitude-indicator', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

            # COL 3, ROW 2: DYNAMICS METRICS
            html.Div([
                html.Div("DYNAMICS METRICS", className='section-label'),
                dcc.Graph(id='dynamics-metrics', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

            # COL 1, ROW 3: G-FORCE
            html.Div([
                html.Div("G-FORCE RADAR", className='section-label'),
                dcc.Graph(id='g-force-meter', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

            # COL 2, ROW 3: WHEEL DYNAMICS
            html.Div([
                html.Div("WHEEL DYNAMICS", className='section-label'),
                dcc.Graph(id='wheel-dynamics', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

            # COL 3, ROW 3: RADAR TARGETS
            html.Div([
                html.Div("RADAR TARGETS", className='section-label'),
                dcc.Graph(id='radar-display', config={'displayModeBar': False}, className='graph-container')
            ], className='grid-item'),

        ], className='dashboard-grid'),

        # Bottom Controls
        html.Div([
            html.Button("PLAY", id='play-btn', className='ctrl-btn'),
            html.Button("PAUSE", id='pause-btn', className='ctrl-btn'),
            html.Div([
                dcc.Slider(0, 100, 1, value=0, id='timeline-slider',
                           className='custom-slider', updatemode='drag')
            ], className='slider-container')
        ], className='bottom-controls')
    ], id='main-container')

# ----------------------------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------------------------
@app.callback(
    [Output('animation-state', 'data'),
     Output('animation-interval', 'disabled')],
    [Input('play-btn', 'n_clicks'),
     Input('pause-btn', 'n_clicks'),
     Input('animation-interval', 'n_intervals'),
     Input('timeline-slider', 'value')],
    [State('animation-state', 'data')]
)
def control_animation(play, pause, interval, slider, state):
    ctx = dash.callback_context
    if not ctx.triggered:
        # On first load, ensure playing is True if desired
        return state, False

    trigger = ctx.triggered[0]['prop_id']
    num_frames = app_data.loader.num_frames if app_data.loader else 1200

    if 'play-btn' in trigger:
        state['playing'] = True
    elif 'pause-btn' in trigger:
        state['playing'] = False
    elif 'animation-interval' in trigger:
        if state['playing']:
            state['frame'] = (state['frame'] + 1) % num_frames
    elif 'timeline-slider' in trigger:
        if not state['playing']:
            state['frame'] = int((slider / 100) * (num_frames - 1))

    return state, not state['playing']

@app.callback(
    [Output('speed-gauge', 'figure'),
     Output('g-force-meter', 'figure'),
     Output('steering-gauge', 'figure'),
     Output('map-display', 'figure'),
     Output('speed-graph', 'figure'),
     Output('time-display', 'children'),
     Output('video-frame-display', 'src'),
     Output('timeline-slider', 'value'),
     Output('vehicle-count-display', 'children'),
     Output('attitude-indicator', 'figure'),
     Output('wheel-dynamics', 'figure'),
     Output('radar-display', 'figure'),
     Output('dynamics-metrics', 'figure'),
     Output('ground-truth-path-display', 'figure')],
    [Input('animation-state', 'data')]
)
def update_view(state):
    frame = state['frame']
    num_frames = app_data.loader.num_frames

    # Get Telemetry Data
    data = app_data.loader.get_frame_data(frame)
    speed_kmh = app_data.speeds[frame]
    speed_ms = speed_kmh / 3.6
    accel = data.get('imu_accel', [0, 0, 0])
    steering = data.get('steering', 0)

    # Video Processing (YOLO + Lane Overlay + ADAS)
    dt = 0.05
    if frame > 0:
        dt = app_data.times[frame] - app_data.times[frame-1]

    img_src = ""
    veh_count = 0
    if app_data.video_proc:
        # Get Annotations (Lanes) for this frame (mapped to % 21)
        xml_anns = app_data.loader.get_frame_annotations(frame % 21)

        # Extract additional ADAS/Mission Control fields from data dictionary
        # data is from get_frame_data which now includes extended metrics
        SteeringAngle = float(data.get('steering', 0))
        DriveState = data.get('drive_state', 'Disengaged')
        TTC = data.get('time_to_collision', None)

        # New Mission Control Features
        Targets = app_data.loader.get_radar_targets(frame)
        GT_Path = app_data.loader.get_ground_truth_path(frame)
        Attitude = app_data.loader.get_attitude(frame)
        Wheels = app_data.loader.get_wheel_dynamics(frame)

        processed_frame, veh_count, detections = app_data.video_proc.process_frame(
            frame,
            speed_ms,
            dt,
            annotations=xml_anns,
            steering_angle=SteeringAngle,
            drive_state=DriveState,
            ttc=TTC,
            radar_targets=Targets,
            ground_truth_path=GT_Path,
            attitude=Attitude,
            wheel_dynamics=Wheels
        )

        if processed_frame is not None:
             _, buffer = cv2.imencode('.jpg', processed_frame)
             encoded = base64.b64encode(buffer).decode('utf-8')
             img_src = f"data:image/jpeg;base64,{encoded}"

    # Map & Gauges
    map_fig = app_data.map_gen.create_map_figure(frame)
    speed_gauge = create_speed_gauge(speed_kmh)
    g_force = create_g_force_meter(accel[0], accel[1])
    steer_gauge = create_steering_gauge(steering)

    # History Graph
    window = 100
    start = max(0, frame - window)
    speed_graph = create_speed_timeline_graph(
        app_data.times[start:frame+1],
        app_data.speeds[start:frame+1],
        app_data.times[frame]
    )

    # New Ultra-Technical Features
    # 1. Attitude Indicator
    attitude_fig = create_attitude_indicator(
        Attitude.get('pitch_deg', 0),
        Attitude.get('roll_deg', 0),
        Attitude.get('yaw_deg', None)
    )

    # 2. Wheel Dynamics
    wheel_speeds = Wheels.get('wheel_speeds', {})
    slip_indicators = Wheels.get('slip', {})
    wheel_fig = create_wheel_dynamics_display(wheel_speeds, slip_indicators)

    # 3. Radar Display
    radar_fig = create_radar_display(Targets)

    # 4. Dynamics Metrics
    dynamics_fig = create_dynamics_metrics_gauge(
        Attitude.get('yaw_rate', 0),
        SteeringAngle,
        speed_kmh,
        data.get('curvature_radius', None)
    )

    # 5. Ground Truth Path
    gt_path_fig = create_ground_truth_path_display(GT_Path)

    time_str = f"T+{app_data.times[frame] - app_data.times[0]:.2f}s | FRAME {frame} / {num_frames}"
    slider_val = (frame / (num_frames - 1)) * 100
    count_str = f"VEHICLES DETECTED: {veh_count}"

    return (speed_gauge, g_force, steer_gauge, map_fig, speed_graph, time_str, 
            img_src, slider_val, count_str, attitude_fig, wheel_fig, 
            radar_fig, dynamics_fig, gt_path_fig)

if __name__ == '__main__':
    load_and_process_data()
    app.layout = create_layout()
    app.run(debug=False, port=8050)
    speed_timeline_graph(
        app_data.times[start:frame+1],
        app_data.speeds[start:frame+1],
        app_data.times[frame]
    )

    # New Ultra-Technical Features
    # 1. Attitude Indicator
    attitude_fig = create_attitude_indicator(
        Attitude.get('pitch_deg', 0),
        Attitude.get('roll_deg', 0),
        Attitude.get('yaw_deg', None)
    )

    # 2. Wheel Dynamics
    wheel_speeds = Wheels.get('wheel_speeds', {})
    slip_indicators = Wheels.get('slip', {})
    wheel_fig = create_wheel_dynamics_display(wheel_speeds, slip_indicators)

    # 3. Radar Display
    radar_fig = create_radar_display(Targets)

    # 4. Dynamics Metrics
    dynamics_fig = create_dynamics_metrics_gauge(
        Attitude.get('yaw_rate', 0),
        SteeringAngle,
        speed_kmh,
        data.get('curvature_radius', None)
    )

    # 5. Ground Truth Path
    gt_path_fig = create_ground_truth_path_display(GT_Path)

    time_str = f"T+{app_data.times[frame] - app_data.times[0]:.2f}s | FRAME {frame} / {num_frames}"
    slider_val = (frame / (num_frames - 1)) * 100
    count_str = f"VEHICLES DETECTED: {veh_count}"

    return (speed_gauge, g_force, steer_gauge, map_fig, speed_graph, time_str, 
            img_src, slider_val, count_str, attitude_fig, wheel_fig, 
            radar_fig, dynamics_fig, gt_path_fig)

if __name__ == '__main__':
    load_and_process_data()
    app.layout = create_layout()
    app.run(debug=False, port=8050)
