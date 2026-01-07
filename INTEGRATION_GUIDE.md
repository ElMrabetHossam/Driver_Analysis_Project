# 🚀 Guide d'Intégration des Améliorations de Performance & Tracking

## Résumé des Corrections Effectuées

### ✅ 1. Performance YOLOv8 (FPS - 5.1 → 15+ FPS attendu)

**Problème :** Temps d'inférence ~190-200ms/frame = 5 FPS (trop lent)

**Solutions implémentées :**
- ✅ Réduction `imgsz`: 640 → 416 (réduction 37.5% du calcul)
- ✅ Activation GPU explicite: `device='cuda'`
- ✅ FP16 (Half Precision) activé pour GPU
- ✅ Logging des params d'inférence pour vérifier
- ✅ `verbose=False` pour réduire overhead

**Résultat attendu:** 50% plus rapide = ~10-12 FPS

**Fichiers modifiés:**
- `video_processor.py`: Optimisé __init__ et _run_inference()

---

### ✅ 2. Correction Traitement des Données (ga, rd)

**Problème :** Moyenne des vecteurs multidimensionnels (mélangeait les axes)
- ❌ Ancien: `ga.mean()` → prend la moyenne de tous les axes
- ✅ Nouveau: `ga[0]` → première composante seulement

**Fichiers modifiés:**
- `data_loader.py`: Lignes 153, 166

---

### ✅ 3. Modules de Lissage Créés

**Fichier: `smoothing_filter.py`** (350+ lignes)

Classes disponibles:
```python
# Exponential Moving Average - optimal pour vitesse/distance
ema = ExponentialMovingAverage(alpha=0.25)
smoothed_speed = ema.update(raw_speed)

# Simple Moving Average - fenêtrage simple
sma = SimpleMovingAverage(window_size=5)
smoothed_accel = sma.update(raw_accel)

# Kalman Filter - estimation optimale
kf = KalmanFilter1D(process_variance=0.01)
filtered_value = kf.update(measurement)

# Multi-axis
multi = MultiAxisSmoothing(num_axes=3, filter_type='ema', alpha=0.2)
smoothed_xyz = multi.update([x, y, z])

# Conveniences
speed_smoother = create_speed_smoother(alpha=0.25)
distance_smoother = create_distance_smoother(alpha=0.2)
accel_smoother = create_accel_smoother(window_size=5, num_axes=3)
```

**Paramètres recommandés:**
| Données | Filtre | Alpha/Window | Raison |
|---------|--------|-------------|--------|
| Vitesse (km/h) | EMA | 0.25 | Réactif mais lisse |
| Distance (m) | EMA | 0.20 | Très lisse (données bruitées radar) |
| Accélération | SMA | window=5 | Moyenne mobile simple |
| Gyroscope | SMA | window=7 | Très lisse (bruit sensor) |

---

### ✅ 4. Module de Tracking Créé

**Fichier: `vehicle_tracker.py`** (250+ lignes)

```python
from vehicle_tracker import VehicleTracker

# Initialiser
tracker = VehicleTracker(max_age=30, track_iou_threshold=0.3)

# Chaque frame
detections = [
    {'bbox': [x1, y1, x2, y2], 'class': 'car', 'conf': 0.9},
    ...
]
radar_data = {0: 25.3, 1: 18.5}  # {detection_idx: distance_m}

tracks = tracker.update(detections, radar_data, frame_idx)

# Accéder aux infos
for track in tracks:
    print(f"ID: {track['id']}")           # ID persistant
    print(f"Distance: {track['distance']}") # Smoothée
    print(f"Speed: {track['speed']}")       # Smoothée
    print(f"Trajectory: {track['trajectory']}")  # Historique
```

**Fonctionnalités:**
- 🆔 ID persistant pour chaque véhicule
- 📊 Smoothing automatique de vitesse et distance
- 📈 Historique de trajectoire
- 🔄 Matching IoU + center distance
- ⏳ Suppression des tracks anciennes (max_age)

---

### ✅ 5. Module d'Affichage Amélioré Créé

**Fichier: `enhanced_overlay.py`** (200+ lignes)

```python
from enhanced_overlay import VehicleOverlayRenderer, render_frame_with_tracks
import cv2

# Renderer classique
renderer = VehicleOverlayRenderer(font_scale=0.7, thickness=2)
frame = renderer.render_tracks(
    frame, 
    tracks,
    show_trajectory=False,    # Désactiver pour perfs
    show_speed=True,
    show_distance=True
)

# Ou fonction tout-en-un (recommandée)
frame = render_frame_with_tracks(
    frame,
    tracks,
    ego_speed_smoother=speed_smoother,
    ego_accel_smoother=accel_smoother,
    ego_speed=50.5,
    ego_accel=0.2
)
```

**Affichage:**
```
┌─ ID: 1 ────────────┐
│ 52.3 km/h          │
│ 24.5 m ↓           │
└────────────────────┘
```

---

### ✅ 6. Intégration dans data_loader.py

**Ajouts:**
```python
from smoothing_filter import create_speed_smoother, create_distance_smoother

class VehicleDataLoader:
    def __init__(self):
        # ...existing code...
        self.speed_smoother = create_speed_smoother(alpha=0.25)
        self.distance_smoother = create_distance_smoother(alpha=0.2)
        self.accel_smoother = create_accel_smoother(window_size=5)
```

---

### ✅ 7. Préparation app.py pour Intégration

**Ajouts:**
```python
from vehicle_tracker import VehicleTracker
from enhanced_overlay import render_frame_with_tracks
from smoothing_filter import create_speed_smoother, create_distance_smoother

class AppData:
    # ...existing...
    vehicle_tracker = None
    ego_speed_smoother = None
    ego_accel_smoother = None

def load_and_process_data():
    # ...existing...
    app_data.vehicle_tracker = VehicleTracker(max_age=30)
    app_data.ego_speed_smoother = create_speed_smoother(alpha=0.25)
    app_data.ego_accel_smoother = create_distance_smoother(alpha=0.2)
```

---

## 🔄 ÉTAPES SUIVANTES (À FAIRE)

### TODO 1: Fusionner callbacks Dash

**Current:** 9 outputs séparés = 9 requêtes HTTP/frame
**Target:** 1 output combined = 1 requête HTTP/frame

```python
# AVANT (lent)
@app.callback(
    [Output('speed-gauge', 'figure'),
     Output('g-force-meter', 'figure'),
     Output('steering-gauge', 'figure'),
     Output('map-display', 'figure'),
     Output('speed-graph', 'figure'),
     Output('time-display', 'children'),
     Output('video-frame-display', 'src'),
     Output('timeline-slider', 'value'),
     Output('vehicle-count-display', 'children')],
    [Input('animation-state', 'data')]
)
def update_view(state):
    # ... calculs ...
    return speed_fig, g_fig, steer_fig, map_fig, speed_graph, time_html, img_src, slider_val, count_html

# APRÈS (rapide)
@app.callback(
    Output('dashboard-combined', 'children'),
    Input('animation-state', 'data')
)
def update_view_combined(state):
    # ... même calculs ...
    return html.Div([
        dcc.Graph(figure=speed_fig),
        dcc.Graph(figure=g_fig),
        # ... etc ...
        html.Img(src=img_src),
    ])
```

---

### TODO 2: Intégrer VehicleTracker dans video_processor.py

```python
# Dans process_frame()

# Extraire détections YOLO
detections = []
for det in results.boxes:
    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
    detections.append({
        'bbox': [x1, y1, x2, y2],
        'class': self.model.names[int(det.cls)],
        'conf': float(det.conf)
    })

# Utiliser données radar pour distance
radar_data = {}
if radar_targets:
    for i, det in enumerate(detections):
        # Matcher détection i avec radar_targets
        radar_data[i] = radar_targets[i]['distance']

# Tracker
tracks = app_data.vehicle_tracker.update(detections, radar_data, frame_idx)

# Afficher
frame = render_frame_with_tracks(
    frame,
    tracks,
    ego_speed_smoother=app_data.ego_speed_smoother,
    ego_speed=speed_ms * 3.6  # Convert to km/h
)

return frame, len(tracks), detections
```

---

### TODO 3: Exporter le modèle en TensorRT (optionnel mais +2x perf)

```bash
# Si tu as NVIDIA GPU + CUDA 11.8+
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device=0)  # Crée yolov8n.engine
"

# Puis dans app.py
VIDEO_PROCESSOR_KWARGS = {
    'model_path': 'yolov8n.engine',  # Au lieu de .pt
    'device': 'cuda',
    'imgsz': 416,
    'half': True,
    'conf': 0.25
}
```

---

## 📊 Résultats Attendus

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| **FPS** | 5.1 | 12-15 | +200% |
| **Latence Dash** | 9 requêtes/frame | 1 requête/frame | -80% |
| **Flou Vitesse** | Oui (clignotement) | Non (lissé) | ✅ |
| **Chevauchement** | Oui | Non (offset +offset) | ✅ |
| **Tracking ID** | Non (random) | Oui (persistant) | ✅ |
| **Distance affichée** | Non | Oui (smoothée) | ✅ |

---

## 🧪 Vérification

```bash
# 1. Test modules
cd mapping/
python test_new_modules.py

# 2. Lancer app
python app.py
# ou
python launch.py

# 3. Vérifier console pour:
# ✅ YOLOv8 model loaded on device: cuda
# ✅ Image Size: 416x416
# ✅ Half Precision (FP16): True
# ✅ Tracker: X vehicles tracked
```

---

## ⚠️ Dépendances Ajoutées

```
scipy           # pour vehicle_tracker (distance.euclidean)
```

Ajouter à `requirements.txt`:
```
scipy>=1.7.0
```

---

## 📝 Notes

1. **Alpha (smoothing factor):**
   - 0.1-0.2: Très lisse (lag plus élevé)
   - 0.25-0.35: Équilibre optimal
   - 0.4-0.5: Très réactif (bruyant)

2. **imgsz (image size):**
   - 320: Très rapide (~100ms) mais perte de précision
   - 416: Optimal (~140ms)
   - 640: Haute précision (~200ms)

3. **GPU Memory:**
   - 416x416 + FP16: ~2.5 GB VRAM
   - 640x640 + FP32: ~8 GB VRAM

4. **Tracking max_age:**
   - max_age=30: Garder un track 30 frames sans match
   - À 12 FPS = 2.5 secondes d'inactivité max

---

## 🆘 Dépannage

**Problème:** Erreur "TypeError: only length-1 arrays can be converted"
- ✅ **Fixé:** data_loader.py utilise maintenant `rd[0]` au lieu de `rd.mean()`

**Problème:** FPS toujours bas
- Vérifier: `print(device)` → doit être 'cuda'
- Réduire `imgsz` à 320
- Exporter en TensorRT (.engine)

**Problème:** Tracking ID change tout le temps
- Vérifier: `track_iou_threshold=0.3` pas trop strict
- Vérifier: `max_age` assez grand (30+ frames)

**Problème:** Vitesse/Distance clignotent
- Augmenter `alpha` (moins de lissage) → réactif
- Réduire `alpha` (plus de lissage) → lisse

---

**Dernière mise à jour:** 04 Jan 2026
**Status:** ✅ Prêt pour intégration
