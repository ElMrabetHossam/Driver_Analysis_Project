
"""
Advanced Dashboard Components Module
Components for ultra-technical vehicle telemetry visualization
"""
import plotly.graph_objects as go
import numpy as np
import math
from config import COLORS

def create_attitude_indicator(pitch_deg, roll_deg, yaw_deg=None):
    """Create an artificial horizon indicator showing pitch and roll"""
    # Create figure with dark background
    fig = go.Figure()

    # Background circle
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1,
                 fillcolor="#0a0a0a", line=dict(color="#333", width=2))

    # Calculate horizon line based on pitch and roll
    # Convert degrees to radians
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    # Create points for horizon line
    # The horizon line will be rotated based on roll angle
    # and shifted based on pitch angle
    line_len = 2.0
    dx = math.cos(roll_rad) * line_len
    dy = math.sin(roll_rad) * line_len

    # Shift based on pitch (vertical shift)
    pitch_shift = math.sin(pitch_rad) * 0.5  # Scale factor for visualization

    # Add horizon line
    fig.add_shape(type="line", x0=-dx, y0=pitch_shift-dy, x1=dx, y1=pitch_shift+dy,
                 line=dict(color="#00d2ff", width=3))

    # Add reference center mark
    fig.add_shape(type="circle", xref="x", yref="y", x0=-0.05, y0=-0.05, x1=0.05, y1=0.05,
                 fillcolor="#ff3d67", line=dict(color="white", width=1))

    # Add pitch and roll text indicators
    fig.add_annotation(x=0, y=-1.3, text=f"PITCH: {pitch_deg:.1f}° | ROLL: {roll_deg:.1f}°",
                      showarrow=False, font=dict(size=12, color="#00d2ff", family="JetBrains Mono"))

    if yaw_deg is not None:
        fig.add_annotation(x=0, y=-1.5, text=f"YAW: {yaw_deg:.1f}°",
                          showarrow=False, font=dict(size=12, color="#00d2ff", family="JetBrains Mono"))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[-1.2, 1.2], visible=False, fixedrange=True),
        yaxis=dict(range=[-1.7, 1.2], visible=False, fixedrange=True),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        title=dict(text="ARTIFICIAL HORIZON", font=dict(color="white"))
    )

    return fig

def create_radar_display(radar_targets, max_range=150):
    """Create a radar display showing detected objects"""
    fig = go.Figure()

    # Create polar radar background
    # Range circles
    for r in [30, 60, 90, 120, 150]:
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r,
                     line=dict(color="rgba(0, 210, 255, 0.2)", width=1))
        fig.add_annotation(x=0, y=r, text=f"{r}m", showarrow=False,
                          font=dict(size=9, color="#444"), yanchor="bottom")

    # Direction lines
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        angle_rad = math.radians(angle)
        x = math.cos(angle_rad) * max_range
        y = math.sin(angle_rad) * max_range
        fig.add_shape(type="line", x0=0, y0=0, x1=x, y1=y,
                     line=dict(color="rgba(0, 210, 255, 0.1)", width=1))

        # Add direction labels
        label_x = math.cos(angle_rad) * (max_range + 10)
        label_y = math.sin(angle_rad) * (max_range + 10)
        direction = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"][angle // 45]
        fig.add_annotation(x=label_x, y=label_y, text=direction, showarrow=False,
                          font=dict(size=10, color="#444"))

    # Add vehicle at center
    fig.add_shape(type="circle", xref="x", yref="y", x0=-5, y0=-5, x1=5, y1=5,
                 fillcolor="#ff3d67", line=dict(color="white", width=1))

    # Plot radar targets
    if radar_targets:
        for target in radar_targets:
            dist = target.get('dist', 0)
            y_rel = target.get('y_rel', 0)  # Lateral offset
            d_rel = target.get('d_rel', 0)  # Relative speed

            if dist > 0 and dist < max_range:
                # Convert to x,y coordinates (assuming y_rel is lateral)
                x = y_rel  # Lateral is x in our display
                y = dist   # Forward distance is y in our display

                # Determine color based on relative speed
                # Closing (positive d_rel) = red, opening (negative) = green
                color = "#ff3d67" if d_rel > 0 else "#10ac84"

                # Add target
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='diamond',
                               line=dict(width=2, color='white')),
                    hoverinfo='text',
                    text=f"Dist: {dist:.1f}m | Rel Speed: {d_rel:.1f}m/s | Lateral: {y_rel:.1f}m"
                ))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[-max_range-20, max_range+20], visible=False, fixedrange=True),
        yaxis=dict(range=[-20, max_range+20], visible=False, fixedrange=True),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        title=dict(text="RADAR TARGETS", font=dict(color="white"))
    )

    return fig

def create_wheel_dynamics_display(wheel_speeds, slip_indicators):
    """Create a wheel dynamics display showing speeds and slip for each wheel"""
    fig = go.Figure()

    # Create wheel layout (top-down view of car)
    # Car dimensions
    car_length = 1.0
    car_width = 0.6

    # Draw car outline
    fig.add_shape(type="rect", x0=-car_width/2, y0=-car_length/2, x1=car_width/2, y1=car_length/2,
                 line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")

    # Draw wheels
    wheel_width = 0.15
    wheel_length = 0.08

    # Front Left
    fl_speed = wheel_speeds.get('fl', 0) if wheel_speeds else 0
    fl_slip = slip_indicators.get('fl', False) if slip_indicators else False
    fl_color = "#ff3d67" if fl_slip else "#10ac84"

    fig.add_shape(type="rect", x0=-car_width/2-wheel_width, y0=car_length/2-wheel_length/2, 
                 x1=-car_width/2, y1=car_length/2+wheel_length/2,
                 line=dict(color="white", width=1), fillcolor=fl_color)
    fig.add_annotation(x=-car_width/2-wheel_width/2, y=car_length/2, text=f"{fl_speed:.0f}",
                      showarrow=False, font=dict(size=10, color="white"))

    # Front Right
    fr_speed = wheel_speeds.get('fr', 0) if wheel_speeds else 0
    fr_slip = slip_indicators.get('fr', False) if slip_indicators else False
    fr_color = "#ff3d67" if fr_slip else "#10ac84"

    fig.add_shape(type="rect", x0=car_width/2, y0=car_length/2-wheel_length/2, 
                 x1=car_width/2+wheel_width, y1=car_length/2+wheel_length/2,
                 line=dict(color="white", width=1), fillcolor=fr_color)
    fig.add_annotation(x=car_width/2+wheel_width/2, y=car_length/2, text=f"{fr_speed:.0f}",
                      showarrow=False, font=dict(size=10, color="white"))

    # Rear Left
    rl_speed = wheel_speeds.get('rl', 0) if wheel_speeds else 0
    rl_slip = slip_indicators.get('rl', False) if slip_indicators else False
    rl_color = "#ff3d67" if rl_slip else "#10ac84"

    fig.add_shape(type="rect", x0=-car_width/2-wheel_width, y0=-car_length/2-wheel_length/2, 
                 x1=-car_width/2, y1=-car_length/2+wheel_length/2,
                 line=dict(color="white", width=1), fillcolor=rl_color)
    fig.add_annotation(x=-car_width/2-wheel_width/2, y=-car_length/2, text=f"{rl_speed:.0f}",
                      showarrow=False, font=dict(size=10, color="white"))

    # Rear Right
    rr_speed = wheel_speeds.get('rr', 0) if wheel_speeds else 0
    rr_slip = slip_indicators.get('rr', False) if slip_indicators else False
    rr_color = "#ff3d67" if rr_slip else "#10ac84"

    fig.add_shape(type="rect", x0=car_width/2, y0=-car_length/2-wheel_length/2, 
                 x1=car_width/2+wheel_width, y1=-car_length/2+wheel_length/2,
                 line=dict(color="white", width=1), fillcolor=rr_color)
    fig.add_annotation(x=car_width/2+wheel_width/2, y=-car_length/2, text=f"{rr_speed:.0f}",
                      showarrow=False, font=dict(size=10, color="white"))

    # Add direction indicator
    fig.add_annotation(x=0, y=car_length/2+0.15, text="FORWARD", showarrow=False,
                      font=dict(size=10, color="#00d2ff"), yanchor="bottom")

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[-1, 1], visible=False, fixedrange=True),
        yaxis=dict(range=[-1, 1], visible=False, fixedrange=True),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        title=dict(text="WHEEL DYNAMICS", font=dict(color="white"))
    )

    return fig

def create_dynamics_metrics_gauge(yaw_rate, steering_angle, speed, curvature_radius=None):
    """Create a gauge showing vehicle dynamics metrics"""
    fig = go.Figure()

    # Create background
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                 fillcolor="#0a0a0a", line=dict(color="#333", width=1))

    # Yaw Rate indicator
    yaw_rate_normalized = max(-1, min(1, yaw_rate / 30))  # Normalize to -1..1 range (assuming max 30 deg/s)
    yaw_rate_color = "#ff3d67" if abs(yaw_rate) > 15 else "#10ac84"

    fig.add_shape(type="rect", x0=0.1, y0=0.7, x1=0.9, y1=0.8,
                 line=dict(color="white", width=1), fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="rect", x0=0.5, y0=0.7, x1=0.5+yaw_rate_normalized*0.4, y1=0.8,
                 line=dict(color="white", width=1), fillcolor=yaw_rate_color)
    fig.add_annotation(x=0.5, y=0.85, text=f"YAW RATE: {yaw_rate:.1f}°/s",
                      showarrow=False, font=dict(size=10, color="white"))

    # Steering Angle indicator
    steering_normalized = max(-1, min(1, steering_angle / 360))  # Normalize to -1..1 range
    steering_color = "#ff3d67" if abs(steering_angle) > 90 else "#10ac84"

    fig.add_shape(type="rect", x0=0.1, y0=0.5, x1=0.9, y1=0.6,
                 line=dict(color="white", width=1), fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=0.5+steering_normalized*0.4, y1=0.6,
                 line=dict(color="white", width=1), fillcolor=steering_color)
    fig.add_annotation(x=0.5, y=0.65, text=f"STEERING: {steering_angle:.1f}°",
                      showarrow=False, font=dict(size=10, color="white"))

    # Speed indicator
    speed_normalized = max(0, min(1, speed / 150))  # Normalize to 0..1 range (assuming max 150 km/h)
    speed_color = "#ff3d67" if speed > 120 else "#10ac84"

    fig.add_shape(type="rect", x0=0.1, y0=0.3, x1=0.9, y1=0.4,
                 line=dict(color="white", width=1), fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="rect", x0=0.1, y0=0.3, x1=0.1+speed_normalized*0.8, y1=0.4,
                 line=dict(color="white", width=1), fillcolor=speed_color)
    fig.add_annotation(x=0.5, y=0.45, text=f"SPEED: {speed:.1f} km/h",
                      showarrow=False, font=dict(size=10, color="white"))

    # Curvature radius indicator (if available)
    if curvature_radius is not None and curvature_radius < 1000:  # Only show if reasonable value
        radius_normalized = max(0, min(1, 1 - curvature_radius / 1000))  # Inverse: smaller radius = higher value
        radius_color = "#ff3d67" if curvature_radius < 50 else "#10ac84"

        fig.add_shape(type="rect", x0=0.1, y0=0.1, x1=0.9, y1=0.2,
                     line=dict(color="white", width=1), fillcolor="rgba(0,0,0,0)")
        fig.add_shape(type="rect", x0=0.1, y0=0.1, x1=0.1+radius_normalized*0.8, y1=0.2,
                     line=dict(color="white", width=1), fillcolor=radius_color)
        fig.add_annotation(x=0.5, y=0.25, text=f"TURN RADIUS: {curvature_radius:.1f}m",
                          showarrow=False, font=dict(size=10, color="white"))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        title=dict(text="DYNAMICS METRICS", font=dict(color="white"))
    )

    return fig

def create_centrifugal_force_gauge(speed, curvature_radius):
    """Create a gauge showing centrifugal force based on speed and turning radius"""
    fig = go.Figure()

    # Calculate centrifugal force: F = m*v²/r (normalized to Gs)
    # Assuming vehicle mass of 1500kg, but normalizing to Gs for display
    if curvature_radius is not None and curvature_radius > 0 and curvature_radius < 1000:
        speed_ms = speed / 3.6  # Convert km/h to m/s
        centrifugal_g = (speed_ms ** 2) / (curvature_radius * 9.81)  # v²/r / g

        # Clamp to reasonable range
        centrifugal_g = max(0, min(2, centrifugal_g))
    else:
        centrifugal_g = 0

    # Create gauge background
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1,
                 fillcolor="#0a0a0a", line=dict(color="#333", width=2))

    # Add scale markers
    for g in [0.5, 1.0, 1.5]:
        fig.add_shape(type="circle", xref="x", yref="y", x0=-g, y0=-g, x1=g, y1=g,
                     line=dict(color="rgba(0, 210, 255, 0.2)", width=1))
        fig.add_annotation(x=0, y=g, text=f"{g}G", showarrow=False,
                          font=dict(size=9, color="#444"), yanchor="bottom")

    # Add force vector
    fig.add_shape(type="line", x0=0, y0=0, x1=centrifugal_g, y1=0,
                 line=dict(color="#ff3d67", width=3))

    # Add force point
    fig.add_trace(go.Scatter(
        x=[centrifugal_g], y=[0],
        mode='markers',
        marker=dict(size=18, color='#ff3d67', symbol='circle',
                    line=dict(width=3, color='white')),
        hoverinfo='text',
        text=f'Centrifugal Force: {centrifugal_g:.2f}G'
    ))

    # Add text display
    fig.add_annotation(x=0, y=-1.3, text=f"<b>{centrifugal_g:.2f}G</b> CENTRIFUGAL",
                      showarrow=False, font=dict(size=12, color="#ff3d67", family="JetBrains Mono"))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[-2.2, 2.2], visible=False, fixedrange=True),
        yaxis=dict(range=[-1.7, 1.7], visible=False, fixedrange=True),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        title=dict(text="CENTRIFUGAL FORCE", font=dict(color="white"))
    )

    return fig

def create_ground_truth_path_display(ground_truth_path, image_shape=(720, 1280)):
    """Create a visualization of the projected ground truth path"""
    if not ground_truth_path or len(ground_truth_path) < 2:
        return None

    fig = go.Figure()

    # Extract x and y coordinates
    x_coords = [p[0] for p in ground_truth_path]
    y_coords = [p[1] for p in ground_truth_path]

    # Add the path
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='lines',
        line=dict(color="#00ff88", width=3),
        name="Ground Truth Path"
    ))

    # Add points at regular intervals for clarity
    interval = max(1, len(ground_truth_path) // 10)
    for i in range(0, len(ground_truth_path), interval):
        fig.add_trace(go.Scatter(
            x=[x_coords[i]], y=[y_coords[i]],
            mode='markers',
            marker=dict(size=6, color="#00ff88"),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[0, image_shape[1]], visible=False, fixedrange=True),
        yaxis=dict(range=[image_shape[0], 0], visible=False, fixedrange=True),  # Inverted for image coordinates
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=True,
        title=dict(text="GROUND TRUTH PATH", font=dict(color="white"))
    )

    return fig

def create_ground_truth_path_overlay(ground_truth_path):
    """Create a visualization of the ground truth path overlay"""
    fig = go.Figure()

    if ground_truth_path and len(ground_truth_path) > 1:
        # Extract x and y coordinates
        x = [p[0] for p in ground_truth_path]
        y = [p[1] for p in ground_truth_path]

        # Add the path
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='#00d2ff', width=3),
            fill='tozeroy', 
            fillcolor='rgba(0, 210, 255, 0.1)'
        ))

        # Add points at specific intervals
        interval = max(1, len(ground_truth_path) // 10)  # Show at most 10 points
        for i in range(0, len(ground_truth_path), interval):
            fig.add_trace(go.Scatter(
                x=[x[i]], y=[y[i]],
                mode='markers',
                marker=dict(size=6, color='white', symbol='diamond')
            ))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        title=dict(text="GROUND TRUTH PATH", font=dict(color="white"))
    )

    return fig
