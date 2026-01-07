"""
Dynamic Map Generator Module
Creates interactive Plotly maps using OpenStreetMap tiles.
Supports bearing, pitch (isometric view), and path highlighting.
"""
import plotly.graph_objects as go
import numpy as np
from config import COLORS

class DynamicMapGenerator:
    """Generates interactive maps for vehicle trajectory using OpenStreetMap"""
    
    def __init__(self, gps_coords, headings):
        """
        Initialize dynamic map generator
        Args:
            gps_coords: numpy array (n, 3) - lat, lon, alt
            headings: numpy array (n,) - orientation in degrees
        """
        self.gps_coords = gps_coords
        self.headings = headings
        self.num_points = len(gps_coords)
        
    def create_map_figure(self, current_frame=0):
        """
        Create a Mapbox figure using OpenStreetMap tiles.
        """
        if current_frame < 0: current_frame = 0
        if self.num_points > 0 and current_frame >= self.num_points:
            current_frame = self.num_points - 1

        lats = self.gps_coords[:, 0]
        lons = self.gps_coords[:, 1]
        cur_lat = lats[current_frame] if self.num_points > 0 else 0
        cur_lon = lons[current_frame] if self.num_points > 0 else 0
        cur_heading = self.headings[current_frame] if self.num_points > 0 else 0

        fig = go.Figure()

        # 1. Full Trajectory (Ghost Path)
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode='lines',
            line=dict(width=2, color='rgba(74, 144, 226, 0.3)'),
            hoverinfo='skip', name='Route'
        ))

        # 2. Traveled Path (Highlighted)
        if current_frame > 0:
            fig.add_trace(go.Scattermapbox(
                lat=lats[:current_frame+1], lon=lons[:current_frame+1],
                mode='lines', line=dict(width=5, color='#FFD700'),
                hoverinfo='skip', name='Path'
            ))

        # 3. Start/End Markers
        if self.num_points > 0:
            fig.add_trace(go.Scattermapbox(
                lat=[lats[0]], lon=[lons[0]], mode='markers',
                marker=dict(size=12, color='#10AC84'), name='Start'
            ))
            fig.add_trace(go.Scattermapbox(
                lat=[lats[-1]], lon=[lons[-1]], mode='markers',
                marker=dict(size=12, color='#EE5A6F'), name='Finish'
            ))

        # 4. Current Vehicle
        fig.add_trace(go.Scattermapbox(
            lat=[cur_lat], lon=[cur_lon], mode='markers+text',
            marker=dict(size=22, color='#FF6B6B', symbol='circle'),
            text=['🚗'], textfont=dict(size=26, color='white'),
            textposition='middle center', name='Vehicle'
        ))

        # Layout Configuration
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=cur_lat, lon=cur_lon),
                zoom=16,
                bearing=cur_heading,
                pitch=45 # Isometric 3D angle
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=COLORS.get('background', '#1E1E1E'),
            showlegend=False,
            template='plotly_dark'
        )

        return fig
