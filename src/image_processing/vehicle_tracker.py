from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

class VehicleTracker:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"--- Chargement du Tracker Véhicules 3D (Sans Vitesse Affichée) : {model_path} ---")
        self.model = YOLO(model_path)
        # Classes cibles : 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = [2, 3, 5, 7]
        
        # --- CALIBRATION PHYSIQUE ---
        self.KNOWN_WIDTH = 1.8  # Largeur réelle approx (m)
        self.FOCAL_LENGTH = 1150 # Focale
        
        # --- MÉMOIRE ---
        self.track_data = {} 

    def get_distance_and_speed(self, track_id, box_width_px, class_id):
        """
        Calcule la distance et la vitesse (Gardé pour la logique de couleur uniquement).
        """
        if box_width_px == 0: return 0.0, 0.0, "Stable"

        # 1. Calcul Distance
        real_width = self.KNOWN_WIDTH
        if class_id == 3: real_width = 0.6 
        if class_id in [5, 7]: real_width = 2.5 

        distance = (real_width * self.FOCAL_LENGTH) / box_width_px
        current_time = time.time()

        # Init mémoire
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                'dists': deque(maxlen=15),
                'times': deque(maxlen=15),
                'speeds': deque(maxlen=10)
            }
            self.track_data[track_id]['dists'].append(distance)
            self.track_data[track_id]['times'].append(current_time)
            return distance, 0.0, "Stable"

        # Récupération historique
        data = self.track_data[track_id]
        prev_dist = data['dists'][-1]
        prev_time = data['times'][-1]
        
        # Mise à jour
        data['dists'].append(distance)
        data['times'].append(current_time)

        # 2. Calcul Vitesse (Interne)
        dt = current_time - prev_time
        if dt > 0:
            delta_dist = distance - prev_dist
            speed_kmh = (delta_dist / dt) * 3.6 
            data['speeds'].append(speed_kmh)
        
        avg_speed = np.mean(data['speeds']) if len(data['speeds']) > 0 else 0.0
        
        # 3. Direction
        if avg_speed < -3.0: direction = "Approaching"
        elif avg_speed > 3.0: direction = "Receding"
        else: direction = "Stable"

        return distance, avg_speed, direction

    def draw_exact_fit_3d(self, frame, x1, y1, x2, y2, color, class_id, cx_img):
        """ Dessine le cube 3D """
        h_img, w_img = frame.shape[:2]

        # Resserrement
        y2 = int(y2 - (y2 - y1) * 0.05) 
        x1 = int(x1 + (x2 - x1) * 0.02)
        x2 = int(x2 - (x2 - x1) * 0.02)

        w, h = x2 - x1, y2 - y1
        cx_box, cy_box = (x1 + x2) // 2, (y1 + y2) // 2

        # 3D
        vanish_x, vanish_y = w_img // 2, int(h_img * 0.46)
        depth_factor = 0.5 
        if class_id in [5, 7]: depth_factor = 1.5 
        if class_id == 3: depth_factor = 0.3 

        perspective_strength = (y2 - vanish_y) / (h_img * 0.5)
        scale = max(0.2, 1.0 - (0.35 * perspective_strength * depth_factor))

        w_f, h_f = int(w * scale), int(h * scale)
        vec_x, vec_y = vanish_x - cx_box, vanish_y - cy_box

        cx_f = int(cx_box + (vec_x * 0.15 * depth_factor))
        cy_f = int(cy_box + (vec_y * 0.15 * depth_factor))

        x1_f, y1_f = int(cx_f - w_f // 2), int(cy_f - h_f // 2)
        x2_f, y2_f = int(cx_f + w_f // 2), int(cy_f + h_f // 2)
        y2_f = int(y2 - (y2 - y2_f) * 0.7) 

        # Dessin
        p_rear = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        p_front = [(x1_f, y1_f), (x2_f, y1_f), (x2_f, y2_f), (x1_f, y2_f)]

        overlay = frame.copy()
        if cx_box < cx_img:
            side_poly = np.array([p_rear[1], p_rear[2], p_front[2], p_front[1]])
        else:
            side_poly = np.array([p_rear[0], p_rear[3], p_front[3], p_front[0]])
            
        cv2.fillPoly(overlay, [side_poly], color)
        top_poly = np.array([p_rear[0], p_rear[1], p_front[1], p_front[0]])
        cv2.fillPoly(overlay, [top_poly], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1_f, y1_f), (x2_f, y2_f), color, 1)
        for i in range(4):
            cv2.line(frame, p_rear[i], p_front[i], color, 1)

        return x1, y1 

    def detect_and_draw(self, frame):
        """ Renvoie les données (frame, count, data) SANS dessiner """
        results = self.model.track(frame, classes=self.target_classes, persist=True, verbose=False, tracker="bytetrack.yaml")
        vehicle_count = 0
        vehicle_boxes_data = [] 
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                width_px = x2 - x1
                dist, speed, direction = self.get_distance_and_speed(track_id, width_px, cls)
                
                vehicle_boxes_data.append({
                    'box': (x1, y1, x2, y2),
                    'cls': cls,
                    'dist': dist,
                    'speed': speed,
                    'dir': direction
                })
                vehicle_count += 1
        
        return frame, vehicle_count, vehicle_boxes_data

    def draw_only(self, frame, vehicle_boxes_data):
        """
        Dessine les boîtes 3D et l'info (Distance Seulement).
        """
        h_img, w_img = frame.shape[:2]
        cx_img = w_img // 2
        
        for data in vehicle_boxes_data:
            x1, y1, x2, y2 = data['box']
            cls = data['cls']
            dist = data['dist']
            speed = data['speed']
            direction = data['dir']
            
            # Couleur dynamique (basée sur la vitesse calculée en interne)
            if dist < 15: color = (0, 0, 255)
            elif dist < 40 and speed < -10: color = (0, 0, 255)
            elif dist < 40: color = (0, 165, 255)
            else: color = (0, 255, 0)
            
            # Dessin 3D
            tx, ty = self.draw_exact_fit_3d(frame, x1, y1, x2, y2, color, cls, cx_img)
            
            # --- HUD INFO (MODIFIÉ: DISTANCE SEULEMENT) ---
            arrow = "v" if direction == "Approaching" else "^"
            
            # Rectangle plus petit (hauteur 25px au lieu de 35px)
            cv2.rectangle(frame, (tx, ty - 25), (tx + 85, ty), color, -1)
            
            # Texte Distance uniquement (Centré)
            cv2.putText(frame, f"{int(dist)}m {arrow}", (tx+5, ty-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # LA LIGNE DE VITESSE A ÉTÉ SUPPRIMÉE ICI

        return frame, len(vehicle_boxes_data)