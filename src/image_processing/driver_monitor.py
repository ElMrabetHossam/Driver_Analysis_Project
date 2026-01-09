import cv2
import numpy as np
import time

class DriverMonitor:
    def __init__(self):
        print("--- Chargement du Moniteur V5 (Recorder Activated) ---")
        
        # --- 1. PHYSIQUE & LISSAGE ---
        self.lane_center_ema = None 
        self.alpha = 0.15 
        self.lateral_speed = 0
        
        # --- 2. LOGIQUE ETATS ---
        self.state = "STABLE"
        self.lane_change_timer = 0
        self.zigzag_counter = 0
        self.last_dir = 0
        
        # --- 3. SECURITE ---
        self.min_headway = 2.0 
        self.ttc_danger = 1.5 
        
        # --- 4. STATISTIQUES ---
        self.start_time = time.time()
        self.total_dist = 0
        self.incidents = {"ZigZag": 0, "Freinage": 0, "Close": 0}
        self.safety_score = 100.0
        self.frame_count = 0  # <--- AJOUTÉ ICI (La correction)
        
        # --- 5. INTERFACE ---
        self.alert_msg = None
        self.alert_frame_count = 0
        self.alert_color = (0, 0, 0)
        self.prev_speed = 0

        # --- 6. ENREGISTREUR DE DONNÉES (DATA LOGGING) ---
        self.history_speed = []
        self.history_score = []
        self.history_deviation = [] 
        self.history_ttc = []

    def smooth_value(self, current, previous):
        if previous is None: return current
        return (1 - self.alpha) * previous + self.alpha * current

    def update_physics(self, w, lane_poly):
        screen_center = w // 2
        lane_center = screen_center
        valid_lane = False
        if lane_poly is not None:
            pts = lane_poly.reshape(-1, 2)
            pts_y_max = np.max(pts[:, 1])
            relevant_pts = pts[pts[:, 1] > pts_y_max * 0.85]
            if len(relevant_pts) > 0:
                lane_center = int(np.mean(relevant_pts[:, 0]))
                valid_lane = True
        
        raw_deviation = lane_center - screen_center
        self.lane_center_ema = self.smooth_value(raw_deviation, self.lane_center_ema)
        
        if valid_lane:
            new_lat_speed = raw_deviation - (self.lane_center_ema if self.lane_center_ema else 0)
            self.lateral_speed = self.smooth_value(new_lat_speed, self.lateral_speed)
        else:
            self.lateral_speed = 0
        return self.lane_center_ema

    def check_maneuvers(self, speed_kmh):
        if speed_kmh < 20: return
        threshold_lat_speed = 15.0
        
        if abs(self.lateral_speed) > threshold_lat_speed:
            self.lane_change_timer = 40
            self.state = "LANE CHANGE"
            self.zigzag_counter = 0
        elif self.lane_change_timer > 0:
            self.lane_change_timer -= 1
            self.state = "LANE CHANGE"
        else:
            self.state = "STABLE"
            if abs(self.lane_center_ema) > 35 and self.state == "STABLE":
                current_dir = 1 if self.lane_center_ema > 0 else -1
                if current_dir != self.last_dir:
                    self.zigzag_counter += 1
                    self.last_dir = current_dir
                if self.zigzag_counter > 2:
                    self.trigger_alert("ZIG-ZAG DETECTE", 2.0, (0, 165, 255))
                    self.zigzag_counter = 0
            else:
                self.zigzag_counter = max(0, self.zigzag_counter - 0.05)

    def check_collision_risks(self, vehicle_data, speed_kmh):
        if speed_kmh < 10: return 99.0, 99.0
        min_ttc = 99.0
        min_headway = 99.0
        speed_ms = speed_kmh / 3.6
        
        for car in vehicle_data:
            cx_car = (car['box'][0] + car['box'][2]) // 2
            if abs(cx_car - 640) < 300: 
                dist = car['dist']
                rel_speed = car['speed']
                if rel_speed < -1.0: 
                    ttc = dist / (abs(rel_speed) / 3.6)
                    if ttc < min_ttc: min_ttc = ttc
                headway = dist / speed_ms
                if headway < min_headway: min_headway = headway

        if min_ttc < self.ttc_danger:
            self.trigger_alert("COLLISION IMMINENTE", 5.0, (0, 0, 255))
            self.incidents["Close"] += 1
        elif min_headway < 1.0:
            if self.alert_frame_count == 0:
                self.trigger_alert("DISTANCE TROP COURTE", 0.2, (0, 69, 255))

        return min_ttc, min_headway

    def trigger_alert(self, text, penalty, color):
        if self.alert_frame_count == 0:
            self.alert_msg = text
            self.alert_color = color
            self.alert_frame_count = 45
            self.safety_score = max(0, self.safety_score - penalty)
            if "ZIG" in text: self.incidents["ZigZag"] += 1
            if "FREIN" in text: self.incidents["Freinage"] += 1

    def draw_hud(self, frame, speed, ttc, headway):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Fond sombre transparent en haut
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(h * 0.12)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Score
        score_col = (0, 255, 0)
        if self.safety_score < 75: score_col = (0, 255, 255)
        if self.safety_score < 50: score_col = (0, 0, 255)
        
        bar_w = int(w * 0.25)
        cv2.line(frame, (20, 20), (20 + bar_w, 20), (100, 100, 100), 4)
        cv2.line(frame, (20, 20), (20 + int(bar_w * (self.safety_score/100)), 20), score_col, 4)
        cv2.putText(frame, f"SAFETY: {int(self.safety_score)}%", (20, 55), font, 0.7, (255, 255, 255), 2)
        
        # Statut (Badge)
        cx_top = w // 2
        status_col = (50, 50, 50) 
        text_col = (200, 200, 200)
        if self.state == "LANE CHANGE":
            status_col = (0, 150, 200)
            text_col = (0, 0, 0)
        elif self.state == "STABLE":
            status_col = (0, 100, 0)
            text_col = (200, 255, 200)
            
        cv2.rectangle(frame, (cx_top - 100, 10), (cx_top + 100, 45), status_col, -1)
        cv2.rectangle(frame, (cx_top - 100, 10), (cx_top + 100, 45), (150, 150, 150), 1)
        (tw, th), _ = cv2.getTextSize(self.state, font, 0.6, 2)
        cv2.putText(frame, self.state, (cx_top - tw//2, 35), font, 0.6, text_col, 2)
        
        # TTC / Headway
        ttc_txt = "TTC: SAFE"
        ttc_col = (0, 255, 0)
        if ttc < 90:
            ttc_txt = f"TTC: {ttc:.1f}s"
            if ttc < 3.0: ttc_col = (0, 0, 255)
            elif ttc < 5.0: ttc_col = (0, 165, 255)
        
        (tw, th), _ = cv2.getTextSize(ttc_txt, font, 0.7, 2)
        cv2.putText(frame, ttc_txt, (w - tw - 20, 40), font, 0.7, ttc_col, 2)
        
        hw_txt = f"HDW: {headway:.1f}s" if headway < 90 else "HDW: >10s"
        (tw2, th2), _ = cv2.getTextSize(hw_txt, font, 0.5, 1)
        cv2.putText(frame, hw_txt, (w - tw2 - 20, 65), font, 0.5, (200, 200, 200), 1)
        
        # Stats discrètes en bas
        duration = int(time.time() - self.start_time)
        m, s = divmod(duration, 60)
        stats = f"DIST: {self.total_dist:.2f}km | TIME: {m:02d}:{s:02d} | ALERTS: {sum(self.incidents.values())}"
        cv2.putText(frame, stats, (20, h - 20), font, 0.5, (150, 150, 150), 1)

    def draw_popups(self, frame):
        if self.alert_frame_count > 0:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            
            # Clignotement
            if self.alert_frame_count % 10 < 5:
                box_w, box_h = 500, 120
                cv2.rectangle(frame, (cx - box_w//2, cy - box_h//2), (cx + box_w//2, cy + box_h//2), self.alert_color, -1)
                cv2.rectangle(frame, (cx - box_w//2, cy - box_h//2), (cx + box_w//2, cy + box_h//2), (255, 255, 255), 4)
                
                (tw, th), _ = cv2.getTextSize(self.alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                text_x = cx - tw // 2
                text_y = cy + th // 2
                cv2.putText(frame, self.alert_msg, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            self.alert_frame_count -= 1

    def update(self, frame, lane_poly, speed_kmh, vehicle_data):
        h, w = frame.shape[:2]
        self.frame_count += 1
        dt = 0.04
        if speed_kmh > 5: self.total_dist += (speed_kmh / 3600) * dt
        
        # Régénération Score
        if self.alert_frame_count == 0 and self.safety_score < 100:
            self.safety_score = min(100, self.safety_score + 0.01)
            
        # Physique
        self.update_physics(w, lane_poly)
        self.check_maneuvers(speed_kmh)
        
        # Freinage
        accel = speed_kmh - self.prev_speed
        if accel < -12 and speed_kmh > 30:
            self.trigger_alert("FREINAGE BRUSQUE", 2.0, (0, 0, 255))
        self.prev_speed = speed_kmh
        
        # Collision
        ttc, headway = self.check_collision_risks(vehicle_data, speed_kmh)
        
        # --- LOGGING ---
        self.history_speed.append(speed_kmh)
        self.history_score.append(self.safety_score)
        
        dev = 0
        if self.lane_center_ema is not None: dev = self.lane_center_ema
        self.history_deviation.append(dev)
        
        ttc_val = min(10, ttc)
        self.history_ttc.append(ttc_val)

        # Rendu
        self.draw_hud(frame, speed_kmh, ttc, headway)
        self.draw_popups(frame)
        
        return frame