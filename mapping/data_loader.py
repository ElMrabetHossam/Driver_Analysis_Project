"""
Data Loader Module
Loads and validates all numpy data files for vehicle tracking
"""
import numpy as np
import os
import math
from config import DATA_FILES


class VehicleDataLoader:
    """Loads and manages vehicle tracking data from numpy files"""
    
    def __init__(self):
        self.positions = None
        self.velocities = None
        self.orientations = None
        self.times = None
        self.gps_times = None
        self.num_frames = 0
        
    def load_all_data(self):
        """Load all data files and validate consistency"""
        print("Loading vehicle tracking data...")
        
        # Load numpy arrays
        self.positions = np.load(DATA_FILES['positions'])
        self.velocities = np.load(DATA_FILES['velocities'])
        self.orientations = np.load(DATA_FILES['orientations'])
        self.times = np.load(DATA_FILES['times'])
        self.gps_times = np.load(DATA_FILES['gps_times'])
        
        # Validate data
        self._validate_data()
        
        self.num_frames = len(self.times)
        print(f"✓ Successfully loaded {self.num_frames} frames of data")
        print(f"  - Duration: {self.times[-1] - self.times[0]:.2f} seconds")
        print(f"  - Positions shape: {self.positions.shape}")
        print(f"  - Velocities shape: {self.velocities.shape}")
        
        return self
    
    def _validate_data(self):
        """Validate that all data arrays have consistent lengths"""
        lengths = {
            'positions': len(self.positions),
            'velocities': len(self.velocities),
            'orientations': len(self.orientations),
            'times': len(self.times),
            'gps_times': len(self.gps_times)
        }
        
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Data length mismatch: {lengths}")
        
        # Validate shapes
        if self.positions.shape[1] != 3:
            raise ValueError(f"Positions should have 3 columns (X,Y,Z), got {self.positions.shape[1]}")
        
        if self.velocities.shape[1] != 3:
            raise ValueError(f"Velocities should have 3 columns (Vx,Vy,Vz), got {self.velocities.shape[1]}")
    
        if self.velocities.shape[1] != 3:
            raise ValueError(f"Velocities should have 3 columns (Vx,Vy,Vz), got {self.velocities.shape[1]}")
    
    def load_additional_data(self):
        """Load and synchronize IMU and CAN data"""
        print("Loading extended telemetry (IMU, CAN)...")
        base_dir = os.path.join(os.path.dirname(DATA_FILES['positions']), '..', 'processed_log')
        
        # Helper to load and interpolate
        def load_interp(subdir, name, target_times):
            try:
                t = np.load(os.path.join(base_dir, subdir, name, 't'))
                v = np.load(os.path.join(base_dir, subdir, name, 'value'))
                
                # Handle different shapes
                if len(v.shape) == 1:
                    # Scalar data
                    return np.interp(target_times, t, v)
                else:
                    # Vector data (interp each component)
                    result = np.zeros((len(target_times), v.shape[1]))
                    for i in range(v.shape[1]):
                        result[:, i] = np.interp(target_times, t, v[:, i])
                    return result
            except Exception as e:
                print(f"⚠️ Failed to load {subdir}/{name}: {e}")
                return None

        # Load and interpolate to main timeline
        self.imu_accel = load_interp('IMU', 'accelerometer', self.times)
        self.steering = load_interp('CAN', 'steering_angle', self.times)
        self.can_speed = load_interp('CAN', 'speed', self.times)
        self.wheel_speeds = load_interp('CAN', 'wheel_speed', self.times)
        # Additional telemetry (best-effort)
        self.gyro = load_interp('IMU', 'gyro', self.times)
        self.throttle = load_interp('CAN', 'throttle', self.times)
        self.brake_pressure = load_interp('CAN', 'brake_pressure', self.times)
        self.gnss_accuracy = load_interp('GNSS', 'live_gnss_qcom', self.times)
        # Radar / lead vehicle distance (if available under CAN/radar)
        self.radar_distance = load_interp('CAN', 'radar', self.times)

        print(f"✓ Extended telemetry loaded and synchronized")
        if self.imu_accel is not None: print(f"  - IMU available")
        if self.steering is not None: print(f"  - Steering available")
        if self.can_speed is not None: print(f"  - CAN Speed available")
        
    def get_frame_data(self, frame_idx):
        """Get all data for a specific frame"""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames-1}]")
        
        data = {
            'position': self.positions[frame_idx],
            'velocity': self.velocities[frame_idx],
            'orientation': self.orientations[frame_idx],
            'time': self.times[frame_idx],
            'gps_time': self.gps_times[frame_idx]
        }
        
        # Add extended data if available
        if hasattr(self, 'imu_accel') and self.imu_accel is not None:
            data['imu_accel'] = self.imu_accel[frame_idx]
        if hasattr(self, 'steering') and self.steering is not None:
            data['steering'] = self.steering[frame_idx]
        if hasattr(self, 'can_speed') and self.can_speed is not None:
            data['can_speed'] = self.can_speed[frame_idx]
        if hasattr(self, 'wheel_speeds') and self.wheel_speeds is not None:
            data['wheel_speeds'] = self.wheel_speeds[frame_idx]
        # Optional additional fields
        if hasattr(self, 'gyro') and self.gyro is not None:
            data['gyro'] = self.gyro[frame_idx]
        if hasattr(self, 'throttle') and self.throttle is not None:
            # Normalize if needed (assume 0-1 or 0-100), try scale detection
            thr = self.throttle[frame_idx]
            if isinstance(thr, (list, tuple, np.ndarray)) and len(np.array(thr).shape) > 0:
                thr = float(np.array(thr).squeeze())
            data['throttle_pct'] = float(thr)
        if hasattr(self, 'brake_pressure') and self.brake_pressure is not None:
            data['brake_pressure'] = float(self.brake_pressure[frame_idx])
        if hasattr(self, 'gnss_accuracy') and self.gnss_accuracy is not None:
            # gnss_accuracy may be vector; take first component or value
            ga = self.gnss_accuracy[frame_idx]
            try:
                ga = np.array(ga).squeeze()
                if ga.size == 1:
                    ga = float(ga)  # Convert to scalar if it's a single value
                elif ga.size > 1:
                    # Handle the case where ga is an array
                    print(f"⚠️ 'ga' is an array with shape {ga.shape}. Using the mean value.")
                    ga = float(ga.mean())  # Example: Use the mean value of the array
                else:
                    raise ValueError("Unexpected empty array for 'ga'")
            except Exception as e:
                raise TypeError(f"Error processing 'ga': {e}")
            data['gnss_accuracy'] = float(ga)
        if hasattr(self, 'radar_distance') and self.radar_distance is not None:
            # radar entry may be scalar or vector; choose first channel
            rd = self.radar_distance[frame_idx]
            rd = np.array(rd).squeeze()
            if rd.size == 1:
                rd = float(rd)
            elif rd.size > 1:
                print(f"⚠️ 'rd' is an array with shape {rd.shape}. Using the mean value.")
                rd = float(rd.mean())
            else:
                raise ValueError("Unexpected empty array for 'rd'")
            data['distance_to_lead'] = float(rd)

        # Derived fields useful for UI
        # longitudinal_velocity: project velocity onto forward axis (assume velocities are in vehicle frame vx,vy,vz)
        try:
            vx = float(self.velocities[frame_idx][0])
            data['longitudinal_velocity'] = vx
        except Exception:
            data['longitudinal_velocity'] = float(self.velocities[frame_idx][0]) if hasattr(self, 'velocities') else 0.0

        # TTC estimate (simple conservative): distance / speed
        if 'distance_to_lead' in data and data['distance_to_lead'] is not None and data['longitudinal_velocity'] > 0.1:
            data['time_to_collision'] = data['distance_to_lead'] / max(1e-3, data['longitudinal_velocity'])
        else:
            data['time_to_collision'] = None

        # drive_state heuristic
        drive_state = 'Disengaged'
        if 'can_speed' in data and data.get('can_speed', 0) is not None:
            try:
                if float(data.get('can_speed', 0)) > 0.1:
                    drive_state = 'Active'
            except Exception:
                pass
        data['drive_state'] = drive_state

        # curvature radius and lateral acceleration from steering and speed (approximate)
        try:
            steering_angle = float(data.get('steering', 0))
            wheelbase = 2.7  # meters, assumed
            if abs(math.tan(math.radians(steering_angle))) > 1e-6:
                r = wheelbase / math.tan(math.radians(steering_angle))
            else:
                r = float('inf')
            data['curvature_radius'] = r
            v = float(data.get('longitudinal_velocity', 0))
            if r != float('inf') and r != 0:
                data['lat_acceleration'] = (v ** 2) / r
            else:
                data['lat_acceleration'] = 0.0
        except Exception:
            data['curvature_radius'] = None
            data['lat_acceleration'] = None
            
        return data
    
    def get_time_range(self):
        """Get the time range of the data"""
        return {
            'start': self.times[0],
            'end': self.times[-1],
            'duration': self.times[-1] - self.times[0]
        }
    
    def get_data_summary(self):
        """Get summary statistics of the data"""
        return {
            'num_frames': self.num_frames,
            'time_range': self.get_time_range(),
            'position_range': {
                'x': (self.positions[:, 0].min(), self.positions[:, 0].max()),
                'y': (self.positions[:, 1].min(), self.positions[:, 1].max()),
                'z': (self.positions[:, 2].min(), self.positions[:, 2].max())
            },
            'has_extended_data': hasattr(self, 'imu_accel')
        }


    def load_annotations(self):
        """Load XML annotations for frames if available"""
        import xml.etree.ElementTree as ET
        self.annotations = {}
        # XMLs are located in data/raw/comma2k19/scb4
        # We navigate from mapping/data_loader.py -> mapping/-> Project Root -> data/raw/comma2k19/scb4
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'raw', 'comma2k19', 'scb4')
        
        print(f"Loading annotations from {base_dir}...")
        count = 0
        # Use modulo 21 logic as per user instruction "c'est un boucle"
        # We only have 0.xml to 20.xml (21 files)
        num_xml_files = 21
        
        for i in range(self.num_frames):
            xml_idx = i % num_xml_files
            xml_path = os.path.join(base_dir, f"{xml_idx}.xml")
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    boxes = []
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        bndbox = obj.find('bndbox')
                        boxes.append({
                            'name': name,
                            'xmin': int(float(bndbox.find('xmin').text)),
                            'ymin': int(float(bndbox.find('ymin').text)),
                            'xmax': int(float(bndbox.find('xmax').text)),
                            'ymax': int(float(bndbox.find('ymax').text))
                        })
                    self.annotations[i] = boxes
                    count += 1
                except Exception:
                    pass
        print(f"✓ Loaded annotations for {count} frames")

    # -----------------------------
    # Extras: projection, radar, wheels, attitude
    # -----------------------------
    def _load_intrinsics(self, intrinsic_path):
        """Try to load intrinsic matrix saved as .npy otherwise return None."""
        try:
            if intrinsic_path and os.path.exists(intrinsic_path):
                K = np.load(intrinsic_path)
                return K
        except Exception:
            pass
        return None

    def get_ground_truth_path(self, frame_idx, K=None, camera_height=None, img_shape=(720,1280)):
        """Project ECEF positions ahead of vehicle into image coordinates using simple pinhole model.

        Returns list of (u,v) image points for a few future frames (ground-truth path on road).
        If camera intrinsics K not provided, use defaults from config.
        """
        from config import CAMERA

        if K is None:
            K = self._load_intrinsics(CAMERA.get('intrinsic_path'))

        # Fallback to simple focal/principal point
        if K is None:
            f = CAMERA.get('focal_length_px', 910)
            cx, cy = (img_shape[1] / 2.0, img_shape[0])
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

        if camera_height is None:
            camera_height = CAMERA.get('camera_height_m', 1.2)

        # We'll take a short horizon of positions from the current frame forward
        pts_img = []
        # choose up to 30 future frames or to end
        end = min(self.num_frames, frame_idx + 30)
        for i in range(frame_idx, end):
            pos = self.positions[i]  # ECEF or vehicle frame; assume vehicle-forward X
            # For simplicity assume positions are in vehicle coordinates (X forward, Y left, Z up)
            X = pos[0]
            Y = pos[1]
            Z = pos[2] - camera_height
            if X <= 0.01:
                continue
            uv = K @ np.array([Y, X, Z])
            if uv[2] == 0:
                continue
            u = int(uv[0] / uv[2])
            v = int(uv[1] / uv[2])
            # clamp
            if 0 <= u < img_shape[1] and 0 <= v < img_shape[0]:
                pts_img.append((u, v))

        return pts_img

    def get_radar_targets(self, frame_idx):
        """Return radar targets list parsed from processed_log/CAN/radar/value if present.

        Each target: {'dist': m, 'd_rel': m/s, 'y_rel': m, 'rcs': optional}
        """
        base_dir = os.path.join(os.path.dirname(DATA_FILES['positions']), '..', 'processed_log')
        radar_val_path = os.path.join(base_dir, 'CAN', 'radar', 'value')
        radar_t_path = os.path.join(base_dir, 'CAN', 'radar', 't')

        try:
            vals = np.load(radar_val_path)
            times = np.load(radar_t_path)
        except Exception:
            return []

        # Find index nearest to self.times[frame_idx]
        ts = self.times[frame_idx]
        idx = int(np.argmin(np.abs(times - ts)))

        raw = vals[idx]
        targets = []
        # raw may be NxM; assume columns correspond to [dist, d_rel, y_rel, rcs]
        raw = np.atleast_2d(raw)
        for r in raw:
            try:
                dist = float(r[0])
                d_rel = float(r[1]) if r.shape[0] > 1 else 0.0
                y_rel = float(r[2]) if r.shape[0] > 2 else 0.0
                targets.append({'dist': dist, 'd_rel': d_rel, 'y_rel': y_rel})
            except Exception:
                continue

        return targets

    def get_wheel_dynamics(self, frame_idx):
        """Return per-wheel speeds and simple slip indicators if wheel_speeds exist."""
        res = {}
        if hasattr(self, 'wheel_speeds') and self.wheel_speeds is not None:
            ws = self.wheel_speeds[frame_idx]
            # assume ordering [fl, fr, rl, rr]
            try:
                fl, fr, rl, rr = ws
                res['wheel_speeds'] = {'fl': float(fl), 'fr': float(fr), 'rl': float(rl), 'rr': float(rr)}
                mean = np.mean([fl, fr, rl, rr])
                # slip when individual wheel deviates from mean > 10%
                res['slip'] = {k: abs(v - mean) / (abs(mean) + 1e-3) > 0.10 for k, v in res['wheel_speeds'].items()}
            except Exception:
                res['wheel_speeds'] = None
                res['slip'] = None
        return res

    def get_attitude(self, frame_idx):
        """Extract pitch, yaw, roll from frame_orientations if available (assumes quaternion or euler stored).

        Returns dict with pitch_deg, yaw_deg, roll_deg, and yaw_rate (if gyro available).
        """
        res = {}
        try:
            q = self.orientations[frame_idx]
            # orientations could be quaternion [w,x,y,z] or euler
            q = np.array(q)
            if q.size == 4:
                # convert quaternion to euler (deg)
                w, x, y, z = q
                # yaw (z), pitch (y), roll (x) conversion
                ysqr = y * y
                t0 = +2.0 * (w * x + y * z)
                t1 = +1.0 - 2.0 * (x * x + ysqr)
                roll = math.degrees(math.atan2(t0, t1))

                t2 = +2.0 * (w * y - z * x)
                t2 = +1.0 if t2 > +1.0 else t2
                t2 = -1.0 if t2 < -1.0 else t2
                pitch = math.degrees(math.asin(t2))

                t3 = +2.0 * (w * z + x * y)
                t4 = +1.0 - 2.0 * (ysqr + z * z)
                yaw = math.degrees(math.atan2(t3, t4))
            else:
                # assume euler
                roll, pitch, yaw = map(float, q[:3])

            res['pitch_deg'] = pitch
            res['yaw_deg'] = yaw
            res['roll_deg'] = roll
        except Exception:
            res['pitch_deg'] = None
            res['yaw_deg'] = None
            res['roll_deg'] = None

        # yaw_rate from gyro if available
        if hasattr(self, 'gyro') and self.gyro is not None:
            try:
                # gyro expected gx,gy,gz in rad/s or deg/s; assume rad/s and convert
                gz = float(self.gyro[frame_idx][2])
                res['yaw_rate'] = math.degrees(gz)
            except Exception:
                res['yaw_rate'] = None
        else:
            res['yaw_rate'] = None

        # slope percent from pitch
        try:
            p = float(res.get('pitch_deg', 0.0))
            res['slope_percent'] = math.tan(math.radians(p)) * 100.0
        except Exception:
            res['slope_percent'] = None

        return res

    def get_frame_annotations(self, frame_idx):
        """Get annotations for a specific frame"""
        if hasattr(self, 'annotations'):
            return self.annotations.get(frame_idx, [])
        return []


if __name__ == "__main__":
    # Test the data loader
    loader = VehicleDataLoader()
    loader.load_all_data()
    loader.load_additional_data()
    loader.load_annotations()
    
    print("\n=== Data Summary ===")
    summary = loader.get_data_summary()
    print(f"Number of frames: {summary['num_frames']}")
    print(f"Duration: {summary['time_range']['duration']:.2f} seconds")
    print(f"Extended data: {summary['has_extended_data']}")
    
    print("\n=== First Frame ===")
    first_frame = loader.get_frame_data(0)
    print(f"Position (ECEF): {first_frame['position']}")
    
    anns = loader.get_frame_annotations(0)
    print(f"Annotations (Frame 0): {len(anns)} objects")

