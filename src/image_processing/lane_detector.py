import cv2
import numpy as np
import os

class LaneDetector:
    def __init__(self):
        """
        Initialisation du détecteur de lignes.
        Pas de modèle IA ici, c'est de la Vision par Ordinateur classique (OpenCV).
        """
        # Constants for lane deviation calculation
        self.LANE_WIDTH_METERS = 3.7  # Standard lane width
        self.HOOD_CUTOFF = 150  # Pixels to cut from bottom (hood of car)
    
    def _create_roi_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create the Region of Interest polygon for lane detection.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Numpy array of polygon vertices
        """
        return np.array([
            [
                (100, height - self.HOOD_CUTOFF),           # Bas Gauche (au-dessus du capot)
                (width - 100, height - self.HOOD_CUTOFF),   # Bas Droit
                (width // 2 + 50, height // 2 + 60),        # Haut Droit (vers l'horizon)
                (width // 2 - 50, height // 2 + 60)         # Haut Gauche
            ]
        ])
    
    def _detect_lane_lines(self, frame):
        """
        Core lane line detection returning raw line segments.
        
        Args:
            frame: BGR image
            
        Returns:
            Tuple of (left_lines, right_lines, edges, height, width)
        """
        height, width = frame.shape[:2]
        
        # 1. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # 2. ROI mask
        polygons = self._create_roi_mask(height, width)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygons, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 3. Hough transform
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30,
                                minLineLength=20, maxLineGap=20)
        
        if lines is None:
            return [], [], masked_edges, height, width
        
        # 4. Separate left and right lanes by slope
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter horizontal lines
            if abs(slope) < 0.3:
                continue
            
            if slope < 0:  # Left lane (negative slope in image coords)
                left_lines.append((x1, y1, x2, y2))
            else:  # Right lane
                right_lines.append((x1, y1, x2, y2))
        
        return left_lines, right_lines, masked_edges, height, width
    
    def calculate_lane_center(self, frame) -> float:
        """
        Calculate the lateral deviation from lane center.
        
        Args:
            frame: BGR image
            
        Returns:
            Deviation in meters (positive = right of center, negative = left)
            Returns np.nan if lanes cannot be detected
        """
        left_lines, right_lines, _, height, width = self._detect_lane_lines(frame)
        
        if not left_lines or not right_lines:
            return np.nan
        
        # Average the line endpoints
        left_avg = np.mean(left_lines, axis=0)
        right_avg = np.mean(right_lines, axis=0)
        
        # Evaluate at a consistent y position (above hood cutoff)
        y_eval = height - self.HOOD_CUTOFF - 50
        
        # Extrapolate left line to y_eval
        x1, y1, x2, y2 = left_avg
        if y2 - y1 != 0:
            left_slope = (x2 - x1) / (y2 - y1)
            left_x = x1 + left_slope * (y_eval - y1)
        else:
            left_x = x1
        
        # Extrapolate right line to y_eval
        x1, y1, x2, y2 = right_avg
        if y2 - y1 != 0:
            right_slope = (x2 - x1) / (y2 - y1)
            right_x = x1 + right_slope * (y_eval - y1)
        else:
            right_x = x1
        
        # Calculate deviation
        lane_center = (left_x + right_x) / 2
        image_center = width / 2
        deviation_pixels = lane_center - image_center
        
        # Convert to meters
        lane_width_pixels = abs(right_x - left_x)
        if lane_width_pixels > 50:  # Sanity check
            meters_per_pixel = self.LANE_WIDTH_METERS / lane_width_pixels
            return float(deviation_pixels * meters_per_pixel)
        
        return np.nan

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