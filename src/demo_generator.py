"""
Demo Video Generator for Driver Behavior Analysis
Generates annotated demo videos with lane detection, vehicle tracking, and driver monitoring.

Usage:
    python3 src/demo_generator.py --input "path/to/segment" --output demo.mp4 --preview
"""

import cv2
import time
import os
import argparse
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Module Imports ---
from src.image_processing.vehicle_tracker import VehicleTracker
from src.image_processing.lane_detector import LaneDetector
from src.image_processing.environment_scanner import EnvironmentScanner
from src.image_processing.driver_monitor import DriverMonitor
from src.image_processing.report_generator import ReportGenerator


class Speedometer:
    """Calculates vehicle speed from optical flow in bird's-eye view."""
    
    def __init__(self):
        self.prev_gray = None
        self.calibration_factor = 3.5 
        self.ym_per_pix = 30 / 720 
        self.real_fps = 24.0 
        self.current_speed = 0.0
        self.alpha = 0.1 
        self.skip_frames = 4 
        self.frame_counter = 0

    def calculate_speed(self, warped_frame, fps_input):
        if fps_input > 0:
            self.real_fps = fps_input
        current_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        h, w = current_gray.shape
        crop_x1, crop_x2 = int(w * 0.4), int(w * 0.6)
        center_strip = current_gray[:, crop_x1:crop_x2]
        inst_speed = self.current_speed 
        self.frame_counter += 1

        if self.prev_gray is not None and self.frame_counter % self.skip_frames == 0:
            search_h = int(h * 0.9)
            prev_section = self.prev_gray[0:search_h, crop_x1:crop_x2]
            try:
                res = cv2.matchTemplate(center_strip, prev_section, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                pixel_shift = max_loc[1]
                if max_val > 0.8:
                    pixels_per_second = pixel_shift * (self.real_fps / self.skip_frames)
                    meters_per_second = pixels_per_second * self.ym_per_pix
                    raw_speed = meters_per_second * 3.6 * self.calibration_factor
                    if raw_speed < 5:
                        raw_speed = 0 
                    inst_speed = raw_speed
            except:
                pass
            self.prev_gray = current_gray
        elif self.prev_gray is None:
            self.prev_gray = current_gray
        
        self.current_speed = (1 - self.alpha) * self.current_speed + self.alpha * inst_speed
        return int(self.current_speed)


def process_video(input_path: str, output_path: str, max_frames: int = None, show_preview: bool = False):
    """
    Process a video segment with full ADAS analysis.
    
    Args:
        input_path: Path to input video (supports .hevc, .mp4, etc.)
        output_path: Path for output annotated video
        max_frames: Optional limit on frames to process
        show_preview: Whether to show live preview window
    """
    print(f"--- DÉMARRAGE SYSTÈME ADAS ---")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Handle segment path (Comma2k19 format)
    video_file = input_path
    if os.path.isdir(input_path):
        video_file = os.path.join(input_path, "video.hevc")
        if not os.path.exists(video_file):
            print(f"Error: No video.hevc found in {input_path}")
            return None
    
    # 1. Load modules
    lane_det = LaneDetector()
    tracker = VehicleTracker()
    env_scanner = EnvironmentScanner()
    driver_mon = DriverMonitor()
    speedometer = Speedometer()
    reporter = ReportGenerator()
    
    # 2. Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_file}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # 3. Output writer
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("-> Processing...")
    frame_idx = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_idx >= max_frames:
            break

        # --- A. CALCULATIONS ---
        warped_frame, _ = lane_det.get_perspective_transform(frame)
        my_speed = speedometer.calculate_speed(warped_frame, fps)
        frame_lanes, lane_poly = lane_det.detect_lanes(frame)
        _, veh_count, vehicle_data = tracker.detect_and_draw(frame_lanes.copy())
        vehicle_rects = [d['box'] for d in vehicle_data]

        # --- B. DRAWING ---
        frame_env = env_scanner.scan_and_draw(frame_lanes, vehicle_rects, lane_poly)
        final_frame, _ = tracker.draw_only(frame_env, vehicle_data)
        final_frame = driver_mon.update(final_frame, lane_poly, my_speed, vehicle_data)

        out.write(final_frame)
        
        # Progress
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Frame {frame_idx}/{total_frames if total_frames > 0 else '?'} - Elapsed: {elapsed:.1f}s")
        
        # Preview
        if show_preview:
            display = cv2.resize(final_frame, (1280, 720))
            cv2.imshow('ADAS System', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    # --- C. GENERATE REPORT ---
    print("-> Generating safety report...")
    report_name = os.path.splitext(os.path.basename(output_path))[0] + "_report.png"
    reporter.generate_report(driver_mon, report_name)
    
    elapsed = time.time() - start_time
    print(f"--- COMPLETE: {output_path} ({frame_idx} frames in {elapsed:.1f}s) ---")
    
    return {
        'frames_processed': frame_idx,
        'duration_seconds': elapsed,
        'safety_score': driver_mon.safety_score,
        'incidents': driver_mon.incidents
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate annotated demo video with ADAS features")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input video or Comma2k19 segment directory")
    parser.add_argument("--output", "-o", type=str, default="demo_output.mp4",
                        help="Path for output video (default: demo_output.mp4)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process (default: all)")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed', exist_ok=True)
    
    # Run processing
    stats = process_video(
        args.input,
        args.output,
        max_frames=args.max_frames,
        show_preview=args.preview
    )
    
    if stats:
        print(f"\n=== Final Stats ===")
        print(f"Safety Score: {stats['safety_score']:.1f}/100")
        print(f"Incidents: {stats['incidents']}")
