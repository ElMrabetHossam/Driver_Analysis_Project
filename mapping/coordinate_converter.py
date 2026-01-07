"""

Coordinate Converter Module

Converts ECEF (Earth-Centered, Earth-Fixed) coordinates to GPS (latitude, longitude, altitude)

and calculates heading/bearing

"""
import numpy as np
from pyproj import Transformer


class CoordinateConverter:
    """Handles coordinate transformations and heading calculations"""
    
    def __init__(self):
        # Create transformer from ECEF (EPSG:4978) to WGS84 GPS (EPSG:4326)
        self.transformer = Transformer.from_crs(
            "EPSG:4978",  # ECEF
            "EPSG:4326",  # WGS84 (lat, lon, alt)
            always_xy=True
        )
    
    def ecef_to_gps(self, x, y, z):
        """
        Convert ECEF coordinates to GPS
        
        Args:
            x, y, z: ECEF coordinates in meters
            
        Returns:
            tuple: (latitude, longitude, altitude)
        """
        lon, lat, alt = self.transformer.transform(x, y, z)
        return lat, lon, alt
    
    def ecef_array_to_gps(self, positions):
        """
        Convert array of ECEF positions to GPS coordinates
        
        Args:
            positions: numpy array of shape (n, 3) with ECEF coordinates
            
        Returns:
            numpy array of shape (n, 3) with (lat, lon, alt)
        """
        lons, lats, alts = self.transformer.transform(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2]
        )
        
        # Stack into array
        gps_coords = np.column_stack([lats, lons, alts])
        return gps_coords
    
    def calculate_heading(self, lat1, lon1, lat2, lon2):
        """
        Calculate bearing/heading from point 1 to point 2
        
        Args:
            lat1, lon1: Starting point coordinates
            lat2, lon2: Ending point coordinates
            
        Returns:
            float: Heading in degrees (0-360, where 0 is North)
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        diff_lon = np.radians(lon2 - lon1)
        
        # Calculate bearing
        x = np.sin(diff_lon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - \
            np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(diff_lon)
        
        bearing_rad = np.arctan2(x, y)
        bearing_deg = np.degrees(bearing_rad)
        
        # Normalize to 0-360
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    
    def calculate_headings_from_positions(self, gps_coords):
        """
        Calculate headings for all positions based on movement direction
        
        Args:
            gps_coords: numpy array of shape (n, 3) with (lat, lon, alt)
            
        Returns:
            numpy array of headings in degrees
        """
        headings = np.zeros(len(gps_coords))
        
        for i in range(len(gps_coords) - 1):
            lat1, lon1 = gps_coords[i, 0], gps_coords[i, 1]
            lat2, lon2 = gps_coords[i + 1, 0], gps_coords[i + 1, 1]
            headings[i] = self.calculate_heading(lat1, lon1, lat2, lon2)
        
        # Last heading is same as second-to-last
        headings[-1] = headings[-2] if len(headings) > 1 else 0
        
        return headings
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two GPS points using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            float: Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = np.sin(delta_lat / 2) ** 2 + \
            np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def calculate_total_distance(self, gps_coords):
        """
        Calculate total distance traveled
        
        Args:
            gps_coords: numpy array of shape (n, 3) with (lat, lon, alt)
            
        Returns:
            float: Total distance in meters
        """
        total_distance = 0
        for i in range(len(gps_coords) - 1):
            lat1, lon1 = gps_coords[i, 0], gps_coords[i, 1]
            lat2, lon2 = gps_coords[i + 1, 0], gps_coords[i + 1, 1]
            total_distance += self.calculate_distance(lat1, lon1, lat2, lon2)
        
        return total_distance


if __name__ == "__main__":
    # Test coordinate conversion
    converter = CoordinateConverter()
    
    # Test ECEF to GPS
    x, y, z = -2713515.07398321, -4264687.94066501, 3876749.42124279
    lat, lon, alt = converter.ecef_to_gps(x, y, z)
    print(f"ECEF ({x:.2f}, {y:.2f}, {z:.2f})")
    print(f"GPS: Lat={lat:.6f}°, Lon={lon:.6f}°, Alt={alt:.2f}m")
    
    # Test heading calculation
    heading = converter.calculate_heading(37.7749, -122.4194, 37.7849, -122.4094)
    print(f"\nHeading: {heading:.2f}°")
