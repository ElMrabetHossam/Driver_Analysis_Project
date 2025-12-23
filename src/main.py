import cv2
import time
import os
from src.image_processing.vehicle_tracker import VehicleTracker
from src.image_processing.lane_detector import LaneDetector

def process_video(input_path, output_path):
    print(f"--- DÉMARRAGE DU TRAITEMENT ---")
    print(f"Entrée : {input_path}")
    print(f"Sortie : {output_path}")

    # 1. Initialisation des modèles
    print("-> Chargement de YOLO et du Détecteur de Lignes...")
    tracker = VehicleTracker() # Ton script YOLO
    lane_det = LaneDetector()  # Ton script Lignes
    
    # 2. Ouverture Vidéo
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo.")
        return

    # Récupération des infos vidéo pour l'enregistrement
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Création du "Video Writer" pour sauvegarder le résultat
    # Codec 'mp4v' est universel pour le MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    print("-> Traitement en cours (Appuie sur 'q' pour arrêter prématurément)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Fin de la vidéo

        # --- LE PIPELINE EN CASCADE [cite: 57] ---
        
        # A. Détection des Lignes (Sur l'image brute)
        # On le fait d'abord pour que les lignes soient "sous" les boîtes des voitures
        frame_lanes = lane_det.detect_lanes(frame)

        # B. Détection des Véhicules (On passe l'image qui a déjà les lignes)
        final_frame, count = tracker.detect_and_draw(frame_lanes)

        # C. Affichage Info Dashboard (Simulation)
        cv2.putText(final_frame, f"System Status: ACTIVE", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(final_frame, f"Vehicles: {count}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Sauvegarde dans le fichier de sortie
        out.write(final_frame)

        # Affichage à l'écran (optionnel, pour voir ce qui se passe)
        # On réduit la taille pour l'affichage écran uniquement
        display = cv2.resize(final_frame, (1024, 768))
        cv2.imshow('Driver Analysis System - COMPLETE', display)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Frames traitées : {frame_count}...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Nettoyage
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    duration = time.time() - start_time
    print(f"--- TERMINÉ ---")
    print(f"Vidéo sauvegardée sous : {output_path}")
    print(f"Temps total : {duration:.1f}s")

if __name__ == "__main__":
    # Assure-toi que le dossier 'data/processed' existe pour sauvegarder le résultat
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    # Exécution
    process_video("test.hevc", "data/processed/demo_resultat.mp4")