# 📹 Guide d'Amélioration de la Qualité Vidéo

## 🎯 Problèmes Résolus

### ✅ 1. Qualité Vidéo Mauvaise
- **Avant:** Compression JPEG trop agressive (qualité ~95)
- **Après:** Qualité 100% sans perte perceptible
- **Amélioration:** Ajout de **débruitage, sharpening, contraste amélioré**

### ✅ 2. Vitesse & Distance Flous
- **Avant:** Texte petit, sans fond, positionné mal
- **Après:** 
  - Texte avec **fond semi-transparent** pour lisibilité
  - **Couleurs distinctes:** ID (couleur track), Vitesse (cyan), Distance (magenta)
  - **Anti-aliasing** pour qualité de texte
  - **Positionnement optimisé** (pas de chevauchement)

### ✅ 3. Données Clignotantes
- **Avant:** Valeurs brutes = clignotement constant
- **Après:** **Lissage exponentiable** avec paramètres configurables
  - Vitesse: alpha=0.25 (réactif)
  - Distance: alpha=0.2 (très lisse)

---

## 🔧 Configuration Rapide

### Option 1: Utiliser un Preset (Recommandé)

Ouvre `launch.py` et ajoute au début:

```python
import quality_config
quality_config.apply_preset('balanced')  # ou 'best_quality', 'performance'
```

**Presets disponibles:**

| Preset | Qualité | FPS | GPU | CPU | Usage |
|--------|---------|-----|-----|-----|-------|
| `best_quality` | ⭐⭐⭐⭐⭐ | 8-10 | ✅ | ❌ | Bonne GPU |
| `balanced` | ⭐⭐⭐⭐ | 10-12 | ✅ | ⚠️ | Optimal |
| `performance` | ⭐⭐⭐ | 15-18 | ✅ | ✅ | Pas assez GPU |
| `low_end` | ⭐⭐ | 20-24 | ⚠️ | ✅ | CPU only |

### Option 2: Configuration Manuelle

Édite `mapping/quality_config.py`:

```python
# Qualité JPEG
JPEG_QUALITY = 100  # 85-100 (100 = sans perte)

# Améliorations vidéo
VIDEO_DENOISE = True      # Débruitage (très important!)
VIDEO_SHARPEN = True      # Accentuer détails
VIDEO_CONTRAST = True     # Améliorer contraste

# YOLO Performance
YOLO_IMGSZ = 416          # 320/416/640 (plus petit = plus rapide)
YOLO_FP16 = True          # Half precision (GPU only)
YOLO_DEVICE = 'cuda'      # 'cuda' ou 'cpu'

# Affichage
SHOW_EGO_TELEMETRY = True
SHOW_VEHICLE_ID = True
SHOW_VEHICLE_SPEED = True
SHOW_VEHICLE_DISTANCE = True
```

---

## 📊 Tuning Recommandé par Scénario

### Scénario 1: GPU Puissant (RTX 3070+)
```python
quality_config.apply_preset('best_quality')
```
- Qualité vidéo maximale
- Débruitage + Sharpening + Contraste tous activés
- YOLO 416x416 avec FP16
- **Résultat:** Images très claires, vitesse/distance lisibles

### Scénario 2: GPU Moyen (RTX 2080, RTX 4060)
```python
quality_config.apply_preset('balanced')
```
- Bon compromis qualité/FPS
- Débruitage seul (sharpening optionnel)
- YOLO 416x416
- **Résultat:** Qualité acceptable, 12-15 FPS

### Scénario 3: CPU Seulement
```python
quality_config.apply_preset('low_end')
JPEG_QUALITY = 85
YOLO_IMGSZ = 320
```
- Qualité réduite mais jouable
- Pas de débruitage/sharpening (trop cher)
- YOLO réduit
- **Résultat:** 20-24 FPS, mais moins net

### Scénario 4: Optimize pour Vitesse/Distance Lisibles

Si le problème principal est la **lisibilité du texte**:

```python
# Dans quality_config.py
FONT_SIZE = 1.0          # Augmenter taille
TEXT_THICKNESS = 3       # Épaissir
TEXT_ALPHA = 0.9         # Fond plus opaque

# Garder qualité vidéo
JPEG_QUALITY = 100
VIDEO_DENOISE = True     # Essentiel pour netteté
VIDEO_SHARPEN = True     # Aide les petits textes
VIDEO_CONTRAST = True    # Améliore lisibilité
```

---

## 🔍 Debugging: Vérifier la Configuration

Exécute dans un terminal:

```python
python -c "import quality_config; quality_config.print_config()"
```

Output attendu:
```
============================================================
CURRENT VIDEO QUALITY CONFIGURATION
============================================================
JPEG Quality: 100
Video Denoise: True
Video Sharpen: True
Video Contrast: True
YOLO Image Size: 416x416
YOLO FP16: True
YOLO Device: cuda
============================================================
```

---

## 📈 Performance Monitoring

Active les logs dans `quality_config.py`:

```python
LOG_FPS = True              # Afficher FPS
LOG_INFERENCE_TIME = True   # Temps YOLO
LOG_FRAME_SIZE = True       # Taille JPEG
```

Puis lancer:
```bash
python launch.py
```

Console output:
```
✅ YOLOv8 model loaded on device: cuda
   Image Size: 416x416
   Inference time: 145ms
   FPS: 6.9

Frame 1: Denoise=10ms, Sharpen=5ms, Encode=25ms
        → Total frame: 185ms
```

---

## 🚨 Dépannage

### Problème: Vidéo toujours floue après config

**Solution:**
1. Vérifier que `VIDEO_DENOISE = True` ✅
2. Vérifier que `VIDEO_SHARPEN = True` ✅
3. Vérifier que `JPEG_QUALITY = 100` ✅
4. Vérifier que `YOLO_IMGSZ` n'est pas trop petit (min 320) ✅

Si toujours floue → problème source (vidéo originale mauvaise)

### Problème: Texte (vitesse/distance) toujours mal visible

**Solution:**
1. Augmenter `FONT_SIZE`:
   ```python
   FONT_SIZE = 1.2  # Au lieu de 0.8
   ```

2. Augmenter `TEXT_THICKNESS`:
   ```python
   TEXT_THICKNESS = 3  # Au lieu de 2
   ```

3. Augmenter `TEXT_ALPHA` (fond plus opaque):
   ```python
   TEXT_ALPHA = 0.95  # Au lieu de 0.85
   ```

4. Utiliser le preset `best_quality`:
   ```python
   quality_config.apply_preset('best_quality')
   ```

### Problème: FPS trop bas (< 5 FPS)

**Réduction progressive:**

1. Désactiver CLAHE:
   ```python
   VIDEO_CONTRAST = False  # Sauve ~8ms
   ```

2. Désactiver Sharpening:
   ```python
   VIDEO_SHARPEN = False   # Sauve ~5ms
   ```

3. Réduire YOLO size:
   ```python
   YOLO_IMGSZ = 320        # Sauve ~60ms
   YOLO_FP16 = False       # (CPU)
   ```

4. Utiliser preset performance:
   ```python
   quality_config.apply_preset('performance')
   ```

### Problème: GPU Memory Error

**Solution:**
1. Réduire `YOLO_IMGSZ`:
   ```python
   YOLO_IMGSZ = 320  # Au lieu de 416
   ```

2. Désactiver FP16:
   ```python
   YOLO_FP16 = False
   ```

3. Utiliser CPU:
   ```python
   YOLO_DEVICE = 'cpu'
   ```

---

## 💾 Fichiers de Configuration

| Fichier | Description |
|---------|-------------|
| `quality_config.py` | **Configuration principale** |
| `video_quality_enhancer.py` | Classes d'amélioration vidéo |
| `app.py` | Application Dash (intégrée) |

---

## 📝 Exemple Complet d'Utilisation

### app.py (démarrage)
```python
import quality_config

# Charger preset
quality_config.apply_preset('balanced')

# Ou config manuelle
quality_config.JPEG_QUALITY = 100
quality_config.VIDEO_DENOISE = True
quality_config.SHOW_VEHICLE_SPEED = True

# Lancer app
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
```

### Vérifier config avant de lancer:
```bash
cd mapping/
python quality_config.py
# Output: Configuration actuellement chargée
```

---

## 🎬 Résultat Attendu

### Avant
- Vidéo compressée, bruitée, peu nette
- Texte vitesse/distance petit, mal positionné
- Valeurs clignotantes
- FPS bas (5-8)

### Après (avec config 'balanced')
- Vidéo claire, débruitée, accentuée
- Texte **grand, coloré, bien positionné** (pas de chevauchement)
- Vitesse/Distance **lisses** (pas de clignotement)
- FPS **12-15** (acceptable)

---

## ✅ Checklist de Validation

- [ ] Vidéo pas floue? (Si non → augmenter DENOISE/SHARPEN)
- [ ] Texte vitesse visible? (Si non → augmenter FONT_SIZE)
- [ ] Texte distance visible? (Si non → augmenter TEXT_ALPHA)
- [ ] FPS acceptable? (Si non → réduire YOLO_IMGSZ/VIDEO_DENOISE)
- [ ] IDs restent stables? (Si non → vérifier TRACKER_MAX_AGE)
- [ ] Pas de lag GUI? (Si lag → réduire VIDEO_DENOISE)

---

**Version:** 1.0 (Jan 4, 2026)
**Dernière update:** Configuration complète + presets
