import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import math


class VideoProcessor:
    def __init__(self, video_path, model_path, device=None, imgsz=640, half=True, conf=0.25):
        """VideoProcessor optimized for real-time YOLOv8 inference.

        Args:
            video_path (str): Path to input video file.
            model_path (str): Path to YOLOv8 model (.pt or exported engine).
            device (str|None): 'cuda' or 'cpu'. If None, auto-detect.
            imgsz (int): Square image size for model (320/416/640 recommended).
            half (bool): Use FP16 when available (GPU / engine).
            conf (float): Confidence threshold for detections.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.conf = float(conf)
        self.frame_cache = {}

        # Camera parameters (tune for dataset)
        self.focal_length = 910
        self.car_height_real = 1.5

        # Tracking / detection state
        self.tracks = {}
        self.last_frame_idx = -1
        self._fps_smooth = 0.0
        self._alpha = 0.85  # smoothing factor for FPS

        # Device selection
        if device is None:
            self.device = 'cuda' if YOLO and self._cuda_available() else 'cpu'
        else:
            self.device = device

        # Load model with fallback to engine if present
        self.model = None
        try:
            # If a TensorRT engine is provided, ultralytics will load it
            self.model = YOLO(model_path)
            # set model attrs if supported
            if hasattr(self.model.model, 'to') and self.device == 'cuda':
                # ultralytics handles device but we keep the flag
                pass
        except Exception:
            # Last-resort: try loading an exported engine file with same stem
            engine_path = os.path.splitext(model_path)[0] + '.engine'
            if os.path.exists(engine_path):
                self.model = YOLO(engine_path)
            else:
                raise

        # If half precision requested but not available on CPU, disable
        if self.device == 'cpu' and self.half:
            self.half = False

    def _cuda_available(self):
        # lightweight CUDA check; don't import torch here to avoid heavy deps
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def get_frame(self, frame_idx):
        # cache small number of recent frames to reduce expensive seeks
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]

        # SMART SEEKING: optimized for sequential playback
        if frame_idx != self.last_frame_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.last_frame_idx = frame_idx
        # keep tiny cache of last 3 frames
        if len(self.frame_cache) > 5:
            # remove oldest
            oldest = sorted(self.frame_cache.keys())[0]
            del self.frame_cache[oldest]
        self.frame_cache[frame_idx] = frame
        return frame

    def _run_inference(self, frame):
        """Run YOLO inference using stream=True to avoid building large result lists.

        Returns the first (and only) results object for this frame.
        """
        # ultralytics accepts images (ndarray) directly
        # Use stream=True to get generator
        try:
            gen = self.model.predict(source=frame, stream=True, imgsz=self.imgsz, conf=self.conf, half=self.half)
        except TypeError:
            # older API name
            gen = self.model.predict(frame, stream=True, imgsz=self.imgsz, conf=self.conf, half=self.half)

        # get first result (should be single-frame input)
        first = None
        for res in gen:
            first = res
            break
        return first

    def predict_path(self, frame, steering_angle_deg, speed_ms, horizon_s=2.0, steps=20):
        """Predict a ground-plane path in image coordinates using a simple bicycle model.

        Returns a list of (x,y) image points representing the predicted path.
        This is an approximation suitable for a UI overlay.
        """
        # Simple kinematic bicycle model in vehicle frame -> project to image using a pinhole approx
        # We'll simulate points ahead in vehicle coordinates then project assuming camera center at bottom center
        dt = horizon_s / steps
        x = 0.0
        y = 0.0
        yaw = 0.0
        L = 2.7  # wheelbase

        path_pts = []
        for i in range(steps):
            # steering angle to radians
            delta = math.radians(float(steering_angle_deg))
            # update vehicle state
            if abs(math.tan(delta)) > 1e-6:
                R = L / math.tan(delta)
                omega = speed_ms / R
            else:
                omega = 0.0

            # integrate
            x += speed_ms * math.cos(yaw) * dt
            y += speed_ms * math.sin(yaw) * dt
            yaw += omega * dt

            # Project to image: simple perspective: assume camera at height h, focal length f
            cam_h = 1.2
            f = self.focal_length
            # Vehicle coordinates: forward = +X, lateral = +Y
            X = x
            Y = y
            if X <= 0.01:
                ix, iy = None, None
            else:
                # center image at bottom middle
                img_h, img_w = frame.shape[:2]
                cx = img_w / 2.0
                cy = img_h

                u = cx + (Y * f) / X
                v = cy - (cam_h * f) / X
                ix, iy = int(u), int(v)

            if ix is not None and iy is not None:
                # clamp
                if 0 <= ix < frame.shape[1] and 0 <= iy < frame.shape[0]:
                    path_pts.append((ix, iy))

        return path_pts

    def draw_overlays(self, frame, detections, predicted_path, drive_state=None, ttc=None, **kwargs):
        """Draw overlays: path prediction, highlight lead vehicle, draw state and alerts."""
        img = frame
        h, w = img.shape[:2]
        
        # 1. Artificial Horizon (Attitude)
        attitude = kwargs.get('attitude')
        if attitude and attitude.get('pitch_deg') is not None:
            pitch = attitude['pitch_deg']
            roll = attitude['roll_deg']
            # Simple conversion: pitch moves line up/down, roll rotates it
            # Center of screen
            cx, cy = w // 2, h // 2
            # Pitch shift: approx 10 pixels per degree (tune this)
            pitch_shift = int(pitch * 20) 
            
            # Create a line segment large enough to cover screen when rotated
            line_len = w * 2
            # Initial points centered at (cx, cy + pitch_shift)
            y_horizon = cy + pitch_shift
            
            # Rotate points by roll
            angle_rad = math.radians(roll)
            dx = math.cos(angle_rad) * (line_len // 2)
            dy = math.sin(angle_rad) * (line_len // 2)
            
            p1 = (int(cx - dx), int(y_horizon + dy))
            p2 = (int(cx + dx), int(y_horizon - dy))
            
            cv2.line(img, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{pitch:.1f}deg", (w-80, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 2. Ground Truth Path (Cyan) - from ECEF projection
        gt_path = kwargs.get('ground_truth_path')
        if gt_path and len(gt_path) > 1:
            # Draw as smooth curve
            pts = np.array(gt_path, np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # 3. Predicted Path (Orange/Yellow) - from Steering
        if predicted_path and len(predicted_path) > 1:
            for i in range(len(predicted_path)-1):
                cv2.line(img, predicted_path[i], predicted_path[i+1], (0, 165, 255), 3, lineType=cv2.LINE_AA)
            # add faint filled poly under path for probability band
            pts = np.array(predicted_path + [(predicted_path[-1][0], predicted_path[-1][1]+10)], np.int32)
            overlay = img.copy()
            cv2.polylines(overlay, [pts], isClosed=False, color=(0,165,255), thickness=2)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # 4. Radar Targets (Red Triangles)
        radar_targets = kwargs.get('radar_targets')
        if radar_targets:
            # We don't have exact UV for radar targets unless we fuse with camera.
            # But the user asked to visualize them. 
            # Often radar targets are just distances. Without azimuth, we can't place them horizontally well.
            # Assuming 'y_rel' provides lateral offset (meters).
            # Project (dist, y_rel) to UV using simple pinhole logic if possible.
            cam_h = 1.2
            f = self.focal_length
            cx = w / 2.0
            cy = h * 1.0 # approx horizon?
            
            for t in radar_targets:
                dist = t.get('dist', 0)
                y_rel = t.get('y_rel', 0) # lateral
                if dist > 1:
                    # Simple projection
                    # u = cx + (y_rel * f) / dist
                    # v = cy_from_horizon (approx) - (cam_h * f) / dist
                    # Let's align v relative to horizon (h/2 approx)
                    v = (h/2) + 20 - (cam_h * f) / dist * 0.5 # rudimentary tweak
                    u = cx + (y_rel * f / dist)
                    
                    center = (int(u), int(v))
                    # Draw triangle
                    pts = np.array([
                        [center[0], center[1]-10],
                        [center[0]-8, center[1]+5],
                        [center[0]+8, center[1]+5]
                    ], np.int32)
                    cv2.fillPoly(img, [pts], (0, 0, 255))

        # 5. Lead Vehicle & YOLO Detections
        # Identify lead vehicle (closest by dist)
        lead = None
        min_dist = float('inf')
        for d in detections:
            if 'dist' in d and d['dist'] is not None:
                if d['dist'] < min_dist:
                    min_dist = d['dist']
                    lead = d

        # Draw detections; lead vehicle in red/orange, others spring green
        for det in detections:
            x1, y1, x2, y2 = det['box']
            if det is lead:
                color = (0, 120, 255)  # Orange for lead
                thickness = 3
            else:
                color = (0, 255, 127) # Spring Green
                thickness = 2

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{det.get('id',-1)} {det.get('speed',0):.0f}kmh"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # 6. Drive State
        if drive_state is not None:
            state_color = (0,200,0) if drive_state == 'Active' else (200,200,200)
            cv2.putText(img, f"SYSTEM: {drive_state}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

        # 7. Proximity / TTC
        if ttc is not None:
            if ttc < 2.0:
                # draw big red banner
                cv2.rectangle(img, (0, img.shape[0]-60), (img.shape[1], img.shape[0]), (0,0,200), -1)
                cv2.putText(img, f"!! PROXIMITY ALERT TTC: {ttc:.2f}s !!", (20, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            else:
                cv2.putText(img, f"TTC: {ttc:.2f}s", (20, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

        # 8. Wheel Dynamics (HUD Style) if available
        wheels = kwargs.get('wheel_dynamics')
        if wheels and wheels.get('wheel_speeds'):
            ws = wheels['wheel_speeds']
            # Draw tiny HUD in bottom right
            base_x, base_y = w - 120, h - 150
            # FL, FR, RL, RR layout
            # Draw 4 boxes
            wh_w, wh_h = 20, 40
            gap = 30
            
            # FL
            c = (0, 255, 0) if not wheels.get('slip', {}).get('fl') else (0, 0, 255)
            cv2.rectangle(img, (base_x, base_y), (base_x + wh_w, base_y + wh_h), c, -1)
            # FR
            c = (0, 255, 0) if not wheels.get('slip', {}).get('fr') else (0, 0, 255)
            cv2.rectangle(img, (base_x + gap, base_y), (base_x + gap + wh_w, base_y + wh_h), c, -1)
            # RL
            c = (0, 255, 0) if not wheels.get('slip', {}).get('rl') else (0, 0, 255)
            cv2.rectangle(img, (base_x, base_y + wh_h + 10), (base_x + wh_w, base_y + wh_h * 2 + 10), c, -1)
            # RR
            c = (0, 255, 0) if not wheels.get('slip', {}).get('rr') else (0, 0, 255)
            cv2.rectangle(img, (base_x + gap, base_y + wh_h + 10), (base_x + gap + wh_w, base_y + wh_h * 2 + 10), c, -1)
            
            cv2.putText(img, "WHEELS", (base_x, base_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        return img

    def process_frame(self, frame_idx, ego_speed_ms, dt=0.05, annotations=None, **kwargs):
        """Process a frame: detect vehicles, estimate distance and speed.

        This method keeps the original return signature: (image, vehicle_count, detections)
        where detections is a list of dicts with keys: id, box, dist, speed.
        
        Args:
            frame_idx (int): Frame index
            ego_speed_ms (float): Ego vehicle speed in m/s
            dt (float): Time delta
            annotations (list): List of dicts {'name': 'l'/'r', 'xmin', ...} for lanes
            **kwargs: Additional telemetry: steering_angle, drive_state, ttc
        """
        t0 = time.time()

        frame = self.get_frame(frame_idx)
        if frame is None:
            return None, 0, []

        display_h, display_w = frame.shape[:2]
        target_w = min(self.imgsz, display_w)
        scale = target_w / float(display_w)
        target_h = int(display_h * scale)

        # Draw annotations (Lanes) on the *original* frame before resize/inference
        # This ensures they are part of the visual feed
        if annotations:
            for ann in annotations:
                # l = Left Lane (Yellow), r = Right Lane (Blue)
                color = (0, 255, 255) if ann['name'] == 'l' else (255, 100, 0)
                # Draw thickened lines for visibility
                cv2.rectangle(frame, (ann['xmin'], ann['ymin']), (ann['xmax'], ann['ymax']), color, 3)
                # Add label
                cv2.putText(frame, f"LANE {ann['name'].upper()}", (ann['xmin'], ann['ymin']-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Run streaming inference (fast, low memory)
        result = self._run_inference(frame)

        vehicle_count = 0
        current_tracks = {}
        detections = []

        if result is not None and hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            for box in boxes:
                # get xyxy and optionally id
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
                x1, y1, x2, y2 = xyxy
                h = max(1.0, (y2 - y1))
                dist = (self.focal_length * self.car_height_real) / h

                track_id = -1
                try:
                    if getattr(box, 'id', None) is not None:
                        track_id = int(box.id[0].cpu().numpy())
                except Exception:
                    track_id = -1

                vehicle_count += 1
                if track_id != -1:
                    current_tracks[track_id] = {'dist': dist}

                # For now use ego-speed as proxy; future: compute relative speed from track history
                obj_speed_ms = float(ego_speed_ms)
                obj_speed_kmh = obj_speed_ms * 3.6

                detections.append({
                    'id': track_id,
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'dist': float(dist),
                    'speed': float(obj_speed_kmh)
                })

        # Update tracks
        self.tracks = current_tracks

        # Predict Path
        steering_angle = kwargs.get('steering_angle', 0)
        predicted_path = self.predict_path(frame, steering_angle, ego_speed_ms)

        # Draw Overlays (Path + Vehicles + Alerts)
        drive_state = kwargs.get('drive_state', 'Disengaged')
        ttc = kwargs.get('ttc', None)
        
        # Use simple default overlay if kwargs missing, but ideally we use the rich `draw_overlays`
        result_img = self.draw_overlays(frame.copy(), detections, predicted_path, drive_state, ttc)


        # FPS Calculation (smoothed)
        process_time = time.time() - t0
        inst_fps = 1.0 / process_time if process_time > 0 else 0.0
        if self._fps_smooth == 0.0:
            self._fps_smooth = inst_fps
        else:
            self._fps_smooth = (self._alpha * self._fps_smooth) + ((1 - self._alpha) * inst_fps)

        # Draw FPS Indicator
        cv2.putText(result_img, f"FPS: {self._fps_smooth:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize for display to reduce payload
        result_img_small = cv2.resize(result_img, (target_w, target_h))

        return result_img_small, vehicle_count, detections

    def get_frame_with_detections(self, frame_idx, annotations=None, ground_truth_path=None, 
                                   radar_targets=None, attitude=None, wheel_dynamics=None,
                                   ego_speed_ms=0.0, steering_angle=0.0, drive_state='Disengaged',
                                   ttc=None):
        """Get a processed frame with all detections and overlays for display
        
        Args:
            frame_idx: Frame index
            annotations: Lane annotations
            ground_truth_path: List of (x,y) points for ground truth trajectory
            radar_targets: List of radar target dicts
            attitude: Dict with pitch_deg, roll_deg, yaw_deg
            wheel_dynamics: Dict with wheel_speeds and slip indicators
            ego_speed_ms: Ego vehicle speed in m/s
            steering_angle: Steering angle in degrees
            drive_state: Drive state string
            ttc: Time to collision in seconds
            
        Returns:
            Base64 encoded image string for Dash display
        """
        import base64
        
        # Process the frame
        frame = self.get_frame(frame_idx)
        if frame is None:
            return None
        
        # Draw lane annotations
        if annotations:
            for ann in annotations:
                color = (0, 255, 255) if ann['name'] == 'l' else (255, 100, 0)
                cv2.rectangle(frame, (ann['xmin'], ann['ymin']), (ann['xmax'], ann['ymax']), color, 3)
                cv2.putText(frame, f"LANE {ann['name'].upper()}", (ann['xmin'], ann['ymin']-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Run YOLO inference
        result = self._run_inference(frame)
        
        detections = []
        if result is not None and hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
                x1, y1, x2, y2 = xyxy
                h = max(1.0, (y2 - y1))
                dist = (self.focal_length * self.car_height_real) / h
                
                track_id = -1
                try:
                    if getattr(box, 'id', None) is not None:
                        track_id = int(box.id[0].cpu().numpy())
                except Exception:
                    track_id = -1
                
                obj_speed_kmh = ego_speed_ms * 3.6
                
                detections.append({
                    'id': track_id,
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'dist': float(dist),
                    'speed': float(obj_speed_kmh)
                })
        
        # Predict path from steering
        predicted_path = self.predict_path(frame, steering_angle, ego_speed_ms)
        
        # Draw all overlays
        result_img = self.draw_overlays(
            frame.copy(), 
            detections, 
            predicted_path, 
            drive_state, 
            ttc,
            ground_truth_path=ground_truth_path,
            radar_targets=radar_targets,
            attitude=attitude,
            wheel_dynamics=wheel_dynamics
        )
        
        # Add FPS indicator
        cv2.putText(result_img, f"FPS: {self._fps_smooth:.1f}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode to base64 for Dash
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
