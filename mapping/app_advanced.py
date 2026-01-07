"""
ULTRA-ADVANCED ADAS MISSION CONTROL DASHBOARD
Boeing/Tesla-Level Professional Interface
Features: Ground Truth Path, Multi-Radar, Attitude Indicator, Wheel Dynamics, Advanced Metrics
"""
import os
import sys
import dash
from dash import html, dcc, Input, Output, State, no_update
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Import custom modules
from data_loader import VehicleDataLoader
from coordinate_converter import CoordinateConverter
from metrics_calculator import MetricsCalculator
from dynamic_map_generator import DynamicMapGenerator
from video_processor import VideoProcessor
from dashboard_components import (
    create_speed_gauge, create_g_force_meter, create_steering_gauge
)
from advanced_dashboard_components import (
    create_attitude_indicator, create_radar_display, 
    create_wheel_dynamics_display, create_dynamics_metrics_gauge,
    create_centrifugal_force_gauge, create_ground_truth_path_overlay
)
from config import COLORS, VIDEO_FILE, MODEL_PATH
from realtime_predictor import get_predictor

# Transformer model path
TRANSFORMER_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'transformer_model.pt')

# ============================================================================
# APP INITIALIZATION
# ============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], update_title=None)
app.title = "ADAS Mission Control - Advanced Telemetry System"
server = app.server

# ============================================================================
# DATA CONTAINER
# ============================================================================
class AppData:
    loader = None
    converter = None
    calculator = None
    map_gen = None
    video_proc = None
    predictor = None  # Transformer predictor
    gps_coords = None
    headings = None
    speeds = None
    times = None
    is_ready = False

app_data = AppData()

# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_process_data():
    """Load all data and initialize processors"""
    global app_data
    
    # Load vehicle data
    app_data.loader = VehicleDataLoader()
    app_data.loader.load_all_data()
    app_data.loader.load_additional_data()
    app_data.loader.load_annotations()
    
    # Convert coordinates
    app_data.converter = CoordinateConverter()
    app_data.gps_coords = app_data.converter.ecef_array_to_gps(app_data.loader.positions)
    app_data.headings = app_data.converter.calculate_headings_from_positions(app_data.gps_coords)
    
    # Calculate speeds
    app_data.calculator = MetricsCalculator(
        app_data.loader.velocities, 
        app_data.loader.times,
        app_data.gps_coords
    )
    app_data.speeds = app_data.calculator.get_speeds_kmh()
    app_data.times = app_data.loader.times
    
    # Initialize map generator
    app_data.map_gen = DynamicMapGenerator(app_data.gps_coords, app_data.headings)
    
    # Initialize video processor with YOLO
    print(f"Loading Video: {VIDEO_FILE}")
    print(f"Loading Model: {MODEL_PATH}")
    app_data.video_proc = VideoProcessor(VIDEO_FILE, MODEL_PATH)
    
    # Initialize Transformer predictor
    if os.path.exists(TRANSFORMER_MODEL_PATH):
        print(f"Loading Transformer Model: {TRANSFORMER_MODEL_PATH}")
        app_data.predictor = get_predictor(TRANSFORMER_MODEL_PATH, sequence_length=20)
    else:
        print(f"⚠️ Transformer model not found at {TRANSFORMER_MODEL_PATH}")
        app_data.predictor = None
    
    app_data.is_ready = True

# ============================================================================
# LAYOUT - ULTRA-ADVANCED GRID SYSTEM
# ============================================================================
def create_layout():
    """Create advanced dashboard layout with maximum features"""
    
    return dbc.Container(fluid=True, style={
        'backgroundColor': '#000000',
        'minHeight': '100vh',
        'padding': '10px',
        'fontFamily': 'JetBrains Mono, monospace'
    }, children=[
        
        # ====== HEADER ROW ======
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2([
                        html.I(className="fas fa-satellite-dish", style={'marginRight': '15px'}),
                        "ADAS MISSION CONTROL",
                        html.Span(" - ADVANCED TELEMETRY SYSTEM", style={
                            'fontSize': '0.5em',
                            'color': COLORS['secondary'],
                            'marginLeft': '10px'
                        })
                    ], style={
                        'color': COLORS['primary'],
                        'margin': '0',
                        'textAlign': 'center',
                        'fontWeight': 'bold',
                        'textShadow': f'0 0 20px {COLORS["primary"]}'
                    }),
                    html.Div(id='system-status', children=[
                        html.Span("● SYSTEM ACTIVE", style={'color': '#10ac84', 'fontSize': '0.9em'})
                    ], style={'textAlign': 'center', 'marginTop': '5px'})
                ])
            ], width=12)
        ], style={'marginBottom': '10px'}),
        
        # ====== ALERT BANNER ======
        dbc.Row([
            dbc.Col([
                html.Div(id='safety-alert-banner', style={'marginBottom': '10px'})
            ], width=12)
        ]),
        
        # ====== MAIN DISPLAY ROW (Video + Map) ======
        dbc.Row([
            # Video Feed with YOLO + Ground Truth Path Overlay
            dbc.Col([
                html.Div([
                    html.Div("📹 CAMERA FEED + YOLO DETECTION + GROUND TRUTH PATH", style={
                        'color': COLORS['primary'],
                        'fontSize': '0.9em',
                        'fontWeight': 'bold',
                        'marginBottom': '5px',
                        'textAlign': 'center'
                    }),
                    html.Img(
                        id='video-frame',
                        style={
                            'width': '100%',
                            'height': 'auto',
                            'border': f'2px solid {COLORS["primary"]}',
                            'borderRadius': '5px',
                            'boxShadow': f'0 0 20px {COLORS["primary"]}'
                        }
                    )
                ], style={
                    'backgroundColor': '#0a0a0a',
                    'padding': '10px',
                    'borderRadius': '10px',
                    'border': '1px solid #222'
                })
            ], width=7),
            
            # Map
            dbc.Col([
                html.Div([
                    html.Div("🗺️ GPS TRAJECTORY MAP", style={
                        'color': COLORS['secondary'],
                        'fontSize': '0.9em',
                        'fontWeight': 'bold',
                        'marginBottom': '5px',
                        'textAlign': 'center'
                    }),
                    dcc.Graph(
                        id='map',
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ], style={
                    'backgroundColor': '#0a0a0a',
                    'padding': '10px',
                    'borderRadius': '10px',
                    'border': '1px solid #222'
                })
            ], width=5)
        ], style={'marginBottom': '10px'}),
        
        # ====== ADVANCED TELEMETRY ROW 1 (Attitude, Radar, Wheels, Dynamics) ======
        dbc.Row([
            # Artificial Horizon (Attitude Indicator)
            dbc.Col([
                dcc.Graph(id='attitude-indicator', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3),
            
            # Multi-Target Radar Display
            dbc.Col([
                dcc.Graph(id='radar-display', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3),
            
            # Wheel Dynamics (4 Wheels)
            dbc.Col([
                dcc.Graph(id='wheel-dynamics', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3),
            
            # Dynamics Metrics (Yaw Rate, Steering, Speed, Turn Radius)
            dbc.Col([
                dcc.Graph(id='dynamics-metrics', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3)
        ], style={'marginBottom': '10px'}),
        
        # ====== ADVANCED TELEMETRY ROW 2 (Primary Gauges) ======
        dbc.Row([
            # Speed Gauge
            dbc.Col([
                dcc.Graph(id='speed-gauge', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3),
            
            # G-Force Indicator
            dbc.Col([
                dcc.Graph(id='gforce-indicator', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3),
            
            # Steering Indicator
            dbc.Col([
                dcc.Graph(id='steering-indicator', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3),
            
            # Centrifugal Force Gauge
            dbc.Col([
                dcc.Graph(id='centrifugal-force', config={'displayModeBar': False}, style={'height': '220px'})
            ], width=3)
        ], style={'marginBottom': '10px'}),
        
        # ====== ADVANCED METRICS PANEL ======
        dbc.Row([
            dbc.Col([
                html.Div(id='advanced-metrics', style={
                    'backgroundColor': '#0a0a0a',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'border': '1px solid #222',
                    'color': COLORS['text'],
                    'fontSize': '0.85em'
                })
            ], width=12)
        ], style={'marginBottom': '10px'}),
        
        # ====== CONTROL BAR ======
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Slider(
                        id='time-slider',
                        min=0,
                        max=app_data.loader.num_frames - 1 if app_data.is_ready else 100,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag'
                    )
                ], style={
                    'backgroundColor': '#0a0a0a',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'border': '1px solid #222'
                })
            ], width=12)
        ]),
        
        # Hidden interval for auto-play
        dcc.Interval(id='interval', interval=50, n_intervals=0, disabled=False),
        dcc.Store(id='animation-state', data={'playing': True, 'frame': 0})
    ])

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('animation-state', 'data'),
    Input('interval', 'n_intervals'),
    Input('time-slider', 'value'),
    State('animation-state', 'data')
)
def control_animation(n_intervals, slider_value, state):
    """Control animation state"""
    ctx = dash.callback_context
    
    if not app_data.is_ready:
        return state
    
    # If slider moved manually, update frame
    if ctx.triggered and 'time-slider' in ctx.triggered[0]['prop_id']:
        state['frame'] = slider_value
        return state
    
    # Auto-advance frame
    state['frame'] = (state['frame'] + 1) % app_data.loader.num_frames
    return state

@app.callback(
    [
        Output('video-frame', 'src'),
        Output('map', 'figure'),
        Output('speed-gauge', 'figure'),
        Output('gforce-indicator', 'figure'),
        Output('steering-indicator', 'figure'),
        Output('attitude-indicator', 'figure'),
        Output('radar-display', 'figure'),
        Output('wheel-dynamics', 'figure'),
        Output('dynamics-metrics', 'figure'),
        Output('centrifugal-force', 'figure'),
        Output('advanced-metrics', 'children'),
        Output('time-slider', 'value'),
        Output('system-status', 'children'),
        Output('safety-alert-banner', 'children')
    ],
    Input('animation-state', 'data')
)
def update_dashboard(state):
    """Update all dashboard components"""
    
    if not app_data.is_ready:
        return [no_update] * 14
    
    frame_idx = state['frame']
    
    # ====== Get Frame Data ======
    frame_data = app_data.loader.get_frame_data(frame_idx)
    attitude = app_data.loader.get_attitude(frame_idx)
    radar_targets = app_data.loader.get_radar_targets(frame_idx)
    wheel_data = app_data.loader.get_wheel_dynamics(frame_idx)
    annotations = app_data.loader.get_frame_annotations(frame_idx)
    ground_truth_path = app_data.loader.get_ground_truth_path(frame_idx)
    
    # ====== Process Video Frame with YOLO + Ground Truth Overlay ======
    frame_with_detections = app_data.video_proc.get_frame_with_detections(
        frame_idx, 
        annotations=annotations,
        ground_truth_path=ground_truth_path
    )
    
    # ====== Update Map ======
    map_fig = app_data.map_gen.create_map_figure(current_frame=frame_idx)
    
    # ====== Extract Metrics ======
    speed_kmh = app_data.speeds[frame_idx]
    
    # G-forces from IMU
    if 'imu_accel' in frame_data and frame_data['imu_accel'] is not None:
        ax, ay, az = frame_data['imu_accel']
        gx = ax / 9.81
        gy = ay / 9.81
    else:
        gx, gy = 0, 0
    
    # Steering
    steering_angle = frame_data.get('steering', 0)
    if hasattr(steering_angle, '__len__'):
        steering_angle = float(steering_angle[0])
    else:
        steering_angle = float(steering_angle) if steering_angle is not None else 0
    
    # Attitude
    pitch = attitude.get('pitch_deg', 0) or 0
    roll = attitude.get('roll_deg', 0) or 0
    yaw = attitude.get('yaw_deg', 0) or 0
    yaw_rate = attitude.get('yaw_rate', 0) or 0
    slope_percent = attitude.get('slope_percent', 0) or 0
    
    # Curvature radius
    curvature_radius = frame_data.get('curvature_radius')
    if curvature_radius == float('inf') or curvature_radius is None:
        curvature_radius = 999
    
    # ====== Create Figures ======
    speed_fig = create_speed_gauge(speed_kmh)
    gforce_fig = create_g_force_meter(gx * 9.81, gy * 9.81)
    steering_fig = create_steering_gauge(steering_angle)
    attitude_fig = create_attitude_indicator(pitch, roll, yaw)
    radar_fig = create_radar_display(radar_targets)
    
    # Wheel Dynamics
    wheel_speeds = wheel_data.get('wheel_speeds', {})
    slip_indicators = wheel_data.get('slip', {})
    wheel_fig = create_wheel_dynamics_display(wheel_speeds, slip_indicators)
    
    # Dynamics Metrics
    dynamics_fig = create_dynamics_metrics_gauge(yaw_rate, steering_angle, speed_kmh, curvature_radius)
    
    # Centrifugal Force
    centrifugal_fig = create_centrifugal_force_gauge(speed_kmh, curvature_radius)
    
    # ====== Advanced Metrics Panel ======
    # Calculate derived metrics
    longitudinal_vel = frame_data.get('longitudinal_velocity', 0)
    lateral_accel = frame_data.get('lat_acceleration', 0)
    
    # Time to collision
    ttc = frame_data.get('time_to_collision', None)
    ttc_str = f"{ttc:.1f}s" if ttc is not None and ttc < 20 else "N/A"
    
    # Distance to lead vehicle
    dist_to_lead = frame_data.get('distance_to_lead', None)
    dist_str = f"{dist_to_lead:.1f}m" if dist_to_lead is not None else "N/A"
    
    # Understeering/Oversteering detection
    # Compare yaw_rate (actual) vs expected from steering
    expected_yaw_rate = (longitudinal_vel / max(1, abs(curvature_radius))) * 57.3 if curvature_radius != 999 else 0
    yaw_diff = abs(yaw_rate - expected_yaw_rate)
    handling_state = "NEUTRAL"
    if yaw_diff > 5:
        if yaw_rate < expected_yaw_rate:
            handling_state = "🔴 UNDERSTEER DETECTED"
        else:
            handling_state = "🔴 OVERSTEER DETECTED"
    elif yaw_diff > 2:
        handling_state = "⚠️ SLIP TENDENCY"
    else:
        handling_state = "✅ NEUTRAL"
    
    # ====== Real-time Prediction ======
    alert_banner = None
    prediction_info = "Analyzing..."
    
    if app_data.predictor:
        pred_result = app_data.predictor.predict(
            frame_data, 
            radar_targets, 
            wheel_data
        )
        
        if pred_result['ready']:
            alert = pred_result['alert']
            prediction_info = f"{alert['type'].upper()} ({pred_result['confidence']:.2f})"
            
            # Create Alert Banner if priority is high
            if alert['type'] != 'safe':
                alert_banner = dbc.Alert(
                    [
                        html.H4([html.I(className="fas fa-exclamation-triangle"), f" {alert['severity']} ALERT"], className="alert-heading"),
                        html.P(alert['message']),
                        html.Hr(),
                        html.P(f"Confidence: {alert['confidence']*100:.1f}%", className="mb-0")
                    ],
                    color="danger" if alert['type'] == 'aggressive' else "warning",
                    dismissable=True,
                    style={'border': '2px solid white', 'boxShadow': '0 0 15px red'}
                )
    
    advanced_metrics_content = html.Div([
        html.H5("⚡ ADVANCED VEHICLE DYNAMICS", style={
            'color': COLORS['primary'],
            'marginBottom': '10px',
            'textAlign': 'center'
        }),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Strong("🎯 ATTITUDE:"),
                    html.Br(),
                    f"Pitch: {pitch:.2f}° | Roll: {roll:.2f}° | Yaw: {yaw:.2f}°",
                    html.Br(),
                    f"Slope: {slope_percent:.1f}% | Yaw Rate: {yaw_rate:.2f}°/s"
                ])
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Strong("🔄 DYNAMICS:"),
                    html.Br(),
                    f"Turn Radius: {curvature_radius:.1f}m",
                    html.Br(),
                    f"Lateral Accel: {lateral_accel:.2f} m/s²"
                ])
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Strong("🚗 LEAD VEHICLE:"),
                    html.Br(),
                    f"Distance: {dist_str}",
                    html.Br(),
                    f"TTC: {ttc_str}"
                ])
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Strong("🧠 AI PREDICTION:"),
                    html.Br(),
                    prediction_info,
                    html.Br(),
                    f"Handling: {handling_state}"
                ])
            ], width=3)
        ])
    ])
    
    # System Status
    current_time = app_data.times[frame_idx]
    status_display = html.Div([
        html.Span("● SYSTEM ACTIVE", style={'color': '#10ac84', 'marginRight': '20px'}),
        html.Span(f"Frame: {frame_idx}/{app_data.loader.num_frames}", style={'color': '#00d2ff', 'marginRight': '20px'}),
        html.Span(f"Time: {current_time:.2f}s", style={'color': '#00d2ff'})
    ])
    
    return (
        frame_with_detections,
        map_fig,
        speed_fig,
        gforce_fig,
        steering_fig,
        attitude_fig,
        radar_fig,
        wheel_fig,
        dynamics_fig,
        centrifugal_fig,
        advanced_metrics_content,
        frame_idx,
        status_display,
        alert_banner
    )

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    load_and_process_data()
    app.layout = create_layout()
    app.run(debug=False, port=8051)
