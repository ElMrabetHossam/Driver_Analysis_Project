import cv2
import numpy as np

class EnvironmentScanner:
    def __init__(self):
        print("--- Chargement du Scanner SLAM (Exterieur + Exclusion Route Exacte) ---")
        
        # Paramètres (Densité réduite)
        self.feature_params = dict(maxCorners=1500,      
                                   qualityLevel=0.02,    
                                   minDistance=7,        
                                   blockSize=7)

    def get_roi_mask(self, frame, vehicle_boxes, lane_polygon):
        """
        Crée le masque.
        1. Bandes latérales blanches.
        2. Noir sur les véhicules.
        3. NOIR SUR LA FORME EXACTE DE LA ROUTE (Nouveau).
        """
        h, w = frame.shape[:2]
        mask = np.zeros_like(frame[:, :, 0]) # Image noire
        
        # --- ZONES D'INTÉRÊT GÉNÉRALES ---
        # On définit une zone large à gauche et à droite pour scanner
        # On laisse le centre un peu noir par sécurité, mais la forme exacte fera le travail
        
        # Bande Gauche
        cv2.rectangle(mask, (0, 0), (int(w*0.45), int(h*0.9)), 255, -1)
        # Bande Droite
        cv2.rectangle(mask, (int(w*0.55), 0), (w, int(h*0.9)), 255, -1)
        
        # On retire le ciel (Haut)
        cv2.rectangle(mask, (0, 0), (w, int(h * 0.05)), 0, -1) # Juste le tout haut

        # --- EXCLUSION DES VÉHICULES ---
        for (x1, y1, x2, y2) in vehicle_boxes:
            pad = 10
            cv2.rectangle(mask, (x1-pad, y1-pad), (x2+pad, y2+pad), 0, -1)
            
        # --- EXCLUSION DE LA ROUTE (FORME EXACTE) --- <--- NOUVEAU
        if lane_polygon is not None:
            # On dessine le polygone de la route en NOIR (0)
            # np.int32 est nécessaire pour fillPoly
            cv2.fillPoly(mask, [np.int32(lane_polygon)], 0)
        
        return mask

    def scan_and_draw(self, frame, vehicle_boxes, lane_polygon=None):
        # 1. Conversion en gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Masque avec exclusion Véhicules + Route
        roi_mask = self.get_roi_mask(frame, vehicle_boxes, lane_polygon)
        
        # 3. Détection
        points = cv2.goodFeaturesToTrack(gray, mask=roi_mask, **self.feature_params)

        count = 0
        if points is not None:
            points = np.int32(points)
            count = len(points)
            
            for i in points:
                x, y = i.ravel()
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
        
        # HUD
        cv2.putText(frame, f"ENV POINTS: {count}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return frame