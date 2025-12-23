import cv2
import numpy as np
import os

class LaneDetector:
    def __init__(self):
        """
        Initialisation du détecteur de lignes.
        Pas de modèle IA ici, c'est de la Vision par Ordinateur classique (OpenCV).
        """
        pass

    def detect_lanes(self, frame):
        """
        Pipeline complet de détection :
        Image -> Niveaux de gris -> Canny (Bords) -> Masque (ROI) -> Hough (Lignes)
        """
        # 1. Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Flou Gaussien (Réduit le bruit du grain de la route)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Détection des bords (Canny)
        # 50 et 150 sont des seuils standards pour détecter les contrastes forts (blanc sur bitume)
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. Définition de la Zone d'Intérêt (ROI) - Masquage du capot et du ciel
        height, width = frame.shape[:2]
        
        # On coupe les 150 derniers pixels du bas pour cacher le capot de la voiture
        hood_cutoff = 150 
        
        # Polygone trapézoïdal (forme de la route vue en perspective)
        polygons = np.array([
            [
                (100, height - hood_cutoff),           # Bas Gauche (au-dessus du capot)
                (width - 100, height - hood_cutoff),   # Bas Droit
                (width // 2 + 50, height // 2 + 60),   # Haut Droit (vers l'horizon)
                (width // 2 - 50, height // 2 + 60)    # Haut Gauche
            ]
        ])
        
        masked_edges = self.region_of_interest(edges, polygons)
        
        # 5. Transformée de Hough (Trouver les lignes) - PARAMÈTRES SENSIBLES
        # Rho=1 : Précision au pixel près (plus fin que 2)
        # Threshold=30 : Il suffit de 30 votes pour valider une ligne (plus sensible)
        # minLineLength=20 : Accepte les petits traits (pointillés) de 20px
        # maxLineGap=20 : Si deux traits sont séparés de moins de 20px, on les relie
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, 
                                minLineLength=20, maxLineGap=20)
        
        # 6. Dessin des lignes détectées
        line_image = np.zeros_like(frame)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Dessin en BLEU (BGR: 255, 0, 0) avec épaisseur 3
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # 7. Superposition (Image originale + Lignes bleues)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        
        return combo_image

    def region_of_interest(self, image, polygons):
        """
        Applique un masque noir pour ne garder que la zone définie par 'polygons'.
        """
        mask = np.zeros_like(image)
        # Remplit le polygone de blanc (255)
        cv2.fillPoly(mask, polygons, 255)
        # Opération ET logique pour garder l'intersection
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

# --- BLOCK DE TEST (Exécution directe) ---
if __name__ == "__main__":
    video_path = "test.hevc"
    
    if not os.path.exists(video_path):
        print(f"ERREUR : La vidéo '{video_path}' est introuvable à la racine.")
        exit()

    cap = cv2.VideoCapture(video_path)
    lane_detector = LaneDetector()
    
    print("--- Test Détection de Lignes (Sensible aux pointillés) ---")
    print("Appuie sur 'q' pour quitter.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Appel de la détection
            frame_with_lanes = lane_detector.detect_lanes(frame)
            
            # Affichage redimensionné pour bien voir
            display_frame = cv2.resize(frame_with_lanes, (1024, 768))
            cv2.imshow('Lane Detection - Tuned', display_frame)
        
        except Exception as e:
            print(f"Erreur lors du traitement d'une frame : {e}")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()