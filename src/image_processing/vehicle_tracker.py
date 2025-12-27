from ultralytics import YOLO
import cv2
import time
import os

class VehicleTracker:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialise le modèle YOLO.
        """
        print(f"--- Chargement du modèle YOLO : {model_path} ---")
        self.model = YOLO(model_path)
        # Classes COCO : 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = [2, 3, 5, 7]
        
        # Constants for distance estimation (monocular geometry)
        self.FOCAL_LENGTH = 910      # Approximate focal length in pixels
        self.CAR_WIDTH = 1.8         # Average car width in meters
        self.TRUCK_WIDTH = 2.5       # Average truck width in meters
    
    def detect_vehicles(self, frame, conf=0.3):
        """
        Detect vehicles and return bounding boxes.
        
        Args:
            frame: BGR image
            conf: Confidence threshold
            
        Returns:
            List of (x1, y1, x2, y2, class_id, confidence) tuples
        """
        results = self.model(frame, classes=self.target_classes, conf=conf, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append((x1, y1, x2, y2, class_id, confidence))
        
        return detections
    
    def estimate_distance(self, bbox_width: int, class_id: int = 2) -> float:
        """
        Estimate distance to a vehicle using monocular geometry.
        
        Formula: Distance = (Focal Length * Real Width) / Image Width
        
        Args:
            bbox_width: Width of bounding box in pixels
            class_id: COCO class ID (2=car, 7=truck, etc.)
            
        Returns:
            Estimated distance in meters
        """
        if bbox_width <= 0:
            return float('inf')
        
        # Use truck width for trucks/buses, car width otherwise
        if class_id in [5, 7]:  # Bus or truck
            real_width = self.TRUCK_WIDTH
        else:
            real_width = self.CAR_WIDTH
        
        distance = (self.FOCAL_LENGTH * real_width) / bbox_width
        return float(distance)
    
    def get_lead_vehicle_distance(self, frame) -> float:
        """
        Get the distance to the lead vehicle (closest vehicle ahead).
        
        The lead vehicle is defined as the vehicle closest to the center
        of the image (in the current lane) with the largest bottom y-coordinate
        (closest to camera).
        
        Args:
            frame: BGR image
            
        Returns:
            Distance to lead vehicle in meters, or np.nan if no vehicle ahead
        """
        import numpy as np
        
        detections = self.detect_vehicles(frame)
        
        if not detections:
            return np.nan
        
        height, width = frame.shape[:2]
        image_center_x = width / 2
        center_tolerance = width / 3  # Vehicle must be in center third
        
        lead_vehicle = None
        max_y_bottom = 0  # Larger y = closer to camera
        
        for x1, y1, x2, y2, class_id, conf in detections:
            # Check if vehicle is roughly centered (in our lane)
            box_center_x = (x1 + x2) / 2
            if abs(box_center_x - image_center_x) > center_tolerance:
                continue
            
            # Track the closest vehicle (highest y2 value)
            if y2 > max_y_bottom:
                max_y_bottom = y2
                lead_vehicle = (x1, y1, x2, y2, class_id, conf)
        
        if lead_vehicle is None:
            return np.nan
        
        x1, y1, x2, y2, class_id, _ = lead_vehicle
        bbox_width = x2 - x1
        
        return self.estimate_distance(bbox_width, class_id) 

    def detect_and_draw(self, frame):
        """
        Détecte les véhicules et dessine les boîtes sur l'image.
        """
        # Inference
        results = self.model(frame, classes=self.target_classes, conf=0.3, verbose=False)
        vehicle_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Dessin du rectangle (Vert)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label
                label = f"Veh {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                vehicle_count += 1
                
        return frame, vehicle_count

# --- ZONE DE TEST RAPIDE ---
if __name__ == "__main__":
    # On cherche la vidéo à la racine du projet
    video_path = "test.hevc" 
    
    if not os.path.exists(video_path):
        print(f"ERREUR: La vidéo '{video_path}' n'est pas trouvée à la racine.")
        exit()

    cap = cv2.VideoCapture(video_path)
    tracker = VehicleTracker()
    
    print(f"--- Lancement de l'analyse sur {video_path} ---")
    print("Appuie sur 'q' dans la fenêtre vidéo pour arrêter.")

    prev_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Détection
        frame_annotated, count = tracker.detect_and_draw(frame)
        
        # 2. Calcul FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Affichage des stats sur l'écran
        cv2.putText(frame_annotated, f"FPS: {int(fps)}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_annotated, f"Vehicles: {count}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 3. Affichage final
        # Resize pour que ça rentre sur ton écran si la vidéo est 4K
        display_frame = cv2.resize(frame_annotated, (1280, 720))
        cv2.imshow('YOLO Detection - TEST', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()