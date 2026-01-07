"""
Dashboard Components Module - Professional Edition
Redesigned UI components for the vehicle mission control.
"""
import plotly.graph_objects as go
from config import COLORS

def create_speed_gauge(current_speed, max_speed=140):
    """Modern Speed Gauge with Neon Accent"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_speed,
        number={'suffix': " km/h", 'font': {'size': 24, 'color': 'white', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [None, max_speed], 'tickwidth': 1, 'tickcolor': "#444"},
            'bar': {'color': "#00d2ff"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 80], 'color': "rgba(0, 210, 255, 0.05)"},
                {'range': [80, 120], 'color': "rgba(255, 255, 0, 0.1)"},
                {'range': [120, 140], 'color': "rgba(255, 0, 0, 0.1)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': current_speed
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        font={'color': 'white'}
    )
    return fig

def create_g_force_meter(accel_x, accel_y):
    """Supercar-style Precision G-Meter"""
    limit = 1.5
    # Normalize to Gs (assuming m/s^2 input)
    gx = accel_x / 9.81
    gy = accel_y / 9.81
    
    fig = go.Figure()
    
    # 1. Dark circular base (Solid, no transparency issues)
    fig.add_shape(type="circle", xref="x", yref="y", x0=-limit, y0=-limit, x1=limit, y1=limit,
                 fillcolor="#050505", line=dict(color="#222", width=2))
    
    # 2. Dynamic Radial Grid (Neon Blue)
    for r in [0.5, 1.0]:
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r,
                     line=dict(color="rgba(0, 210, 255, 0.2)", width=1, dash="dash"))
        fig.add_annotation(x=0, y=r, text=f"{r}G", showarrow=False, 
                          font=dict(size=9, color="#444"), yanchor="bottom")
    
    # 3. Crosshair (Subtle)
    fig.add_shape(type="line", x0=-limit, y0=0, x1=limit, y1=0, line=dict(color="#111", width=1))
    fig.add_shape(type="line", x0=0, y0=-limit, x1=0, y1=limit, line=dict(color="#111", width=1))

    # 4. G-Force Vector line (from center to current)
    fig.add_shape(type="line", x0=0, y0=0, x1=gx, y1=gy, 
                 line=dict(color="#00ffaa", width=3))

    # 5. Current G-Marker
    fig.add_trace(go.Scatter(
        x=[gx], y=[gy],
        mode='markers',
        marker=dict(size=18, color='#00ffaa', symbol='circle',
                    line=dict(width=3, color='white')),
        hoverinfo='text',
        text=f'Lateral: {gx:.2f}G | Long: {gy:.2f}G'
    ))
    
    # Digital Readout Overlay
    fig.add_annotation(x=0, y=-1.3, text=f"<b>{abs(gx):.2f}G</b> LAT | <b>{abs(gy):.2f}G</b> LON",
                      showarrow=False, font=dict(size=12, color="#00ffaa", family="JetBrains Mono"))
    
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[-limit, limit], visible=False, fixedrange=True),
        yaxis=dict(range=[-limit, limit], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True),
        paper_bgcolor='#050505', 
        plot_bgcolor='#050505',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False
    )
    return fig

def create_steering_gauge(angle):
    """Arc Steering Indicator"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = angle,
        number = {'suffix': "°", 'font': {'size': 20, 'color': 'white'}},
        gauge = {
            'axis': {'range': [-360, 360], 'tickwidth': 1, 'tickcolor': "#444"},
            'bar': {'color': "#ff3d67"},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [-360, 360], 'color': "rgba(255, 255, 255, 0.05)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'value': angle
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        font={'color': 'white'}
    )
    return fig

def create_speed_timeline_graph(times, speeds, current_time):
    """Historical telemetry graph"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=speeds, mode='lines',
        line=dict(color="#00d2ff", width=1.5),
        fill='tozeroy', fillcolor="rgba(0, 210, 255, 0.05)"
    ))
    fig.add_shape(type="line", x0=current_time, y0=0, x1=current_time, y1=max(speeds) if len(speeds)>0 else 1,
                 line=dict(color="white", width=1, dash="dot"))
    
    fig.update_layout(
        margin=dict(l=30, r=10, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#555', title='TIME'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#555', title='KM/H'),
        showlegend=False,
        height=200
    )
    return fig
