import torch
import cv2
import ultralytics
import sklearn
import os

def check_environment():
    print("--- 🛠️ VÉRIFICATION DU SYSTÈME 🛠️ ---")
    
    # 1. Vérification PyTorch (Pour le Deep Learning futur)
    try:
        print(f"✅ PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🚀 GPU Détecté: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ Pas de GPU détecté. Le Deep Learning sera lent (CPU mode).")
    except ImportError:
        print("❌ CRITIQUE: PyTorch non installé.")

    # 2. Vérification OpenCV (Pour le traitement d'image)
    try:
        print(f"✅ OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("❌ CRITIQUE: OpenCV non installé.")

    # 3. Vérification YOLO (Ultralytics)
    try:
        from ultralytics import YOLO
        print(f"✅ Ultralytics (YOLO) installé.")
        # Petit test de téléchargement du modèle
        print("   ⏳ Test de chargement du modèle YOLOv8n (nano)...")
        model = YOLO('yolov8n.pt') 
        print("   ✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"❌ CRITIQUE: Problème avec YOLO. Erreur: {e}")

    # 4. Vérification des dossiers
    required_folders = ['data/raw', 'data/processed', 'src/models']
    print("\n--- 📂 VÉRIFICATION DES DOSSIERS ---")
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✅ Dossier trouvé: {folder}")
        else:
            print(f"❌ MANQUANT: Crée le dossier '{folder}'")

    print("\n--- FIN DU TEST ---")

if __name__ == "__main__":
    check_environment()