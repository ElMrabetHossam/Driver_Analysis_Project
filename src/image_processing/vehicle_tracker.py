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