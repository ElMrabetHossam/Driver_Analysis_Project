#!/usr/bin/env python3
"""
🚀 QUICK START GUIDE - Lancer le Dashboard Amélioré
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          VEHICLE OS - DASHBOARD WITH ENHANCED VIDEO QUALITY               ║
║                                                                            ║
║  Les problèmes suivants ont été résolus:                                  ║
║  ✅ Qualité vidéo mauvaise → Débruitage + Sharpening + Contraste          ║
║  ✅ Vitesse/Distance flous → Texte avec fond, anti-aliasing, colorés      ║
║  ✅ Données clignotantes → Lissage EMA appliqué                           ║
║  ✅ Configuration complexe → Presets et guide complet                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ ÉTAPE 1: LANCER LE DASHBOARD ───────────────────────────────────────────┐
│                                                                           │
│  Option A: Lancement OPTIMISÉ (Recommandé)                              │
│  $ cd mapping/                                                           │
│  $ python launch_optimized.py                                            │
│                                                                           │
│  → Charge automatiquement preset 'balanced'                              │
│  → Affiche configuration avant lancement                                 │
│  → Meilleur rapport qualité/performance                                  │
│                                                                           │
│  Option B: Lancement Standard                                            │
│  $ cd mapping/                                                           │
│  $ python launch.py                                                      │
│                                                                           │
│  → Édite quality_config.py pour ajuster les paramètres                   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ ÉTAPE 2: ACCÉDER AU DASHBOARD ──────────────────────────────────────────┐
│                                                                           │
│  Ouvre ton navigateur:  http://localhost:8050                            │
│                                                                           │
│  Tu devrais voir:                                                         │
│  ✓ Vidéo claire (pas floue)                                              │
│  ✓ Vitesse en CYAN sous chaque véhicule                                   │
│  ✓ Distance en MAGENTA sous les véhicules                                 │
│  ✓ ID en couleur au top-left de chaque boîte                             │
│  ✓ Données LISSES (pas de cligotement)                                    │
│  ✓ Infos ego (vitesse, accel, steering) en haut-left                      │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ ÉTAPE 3: AJUSTER LA QUALITÉ (SI BESOIN) ────────────────────────────────┐
│                                                                           │
│  Si vidéo trop floue:                                                     │
│  → Édite mapping/quality_config.py                                        │
│  → Change JPEG_QUALITY = 100 (déjà défini)                               │
│  → Change VIDEO_DENOISE = True (déjà défini)                             │
│                                                                           │
│  Si texte pas assez visible:                                              │
│  → Augmente FONT_SIZE de 0.8 à 1.0                                       │
│  → Augmente TEXT_THICKNESS de 2 à 3                                      │
│  → Augmente TEXT_ALPHA de 0.85 à 0.95                                    │
│                                                                           │
│  Si FPS trop bas (< 10):                                                  │
│  → Utilise preset 'performance': quality_config.apply_preset('performance')│
│  → Ou réduis YOLO_IMGSZ de 416 à 320                                     │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ PRESETS DISPONIBLES ────────────────────────────────────────────────────┐
│                                                                           │
│  Pour charger un preset, édite launch_optimized.py ligne 21:              │
│                                                                           │
│  preset_to_load = 'best_quality'    # Qualité maximale (GPU needed)      │
│  preset_to_load = 'balanced'        # Optimal (DÉFAUT) ⭐ RECOMMANDÉ      │
│  preset_to_load = 'performance'     # Plus rapide, moins beau            │
│  preset_to_load = 'low_end'         # CPU compatible, image réduite      │
│                                                                           │
│  Ou dans quality_config.py:                                               │
│  quality_config.apply_preset('balanced')                                 │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ FICHIERS PRINCIPAUX ────────────────────────────────────────────────────┐
│                                                                           │
│  mapping/                                                                 │
│  ├── launch_optimized.py          ← Lancer AVEC presets                  │
│  ├── quality_config.py             ← Configuration (ÉDITER si besoin)     │
│  ├── video_quality_enhancer.py     ← Code d'amélioration vidéo           │
│  ├── test_video_quality.py         ← Tests de validation                  │
│  └── app.py                        ← Application Dash                     │
│                                                                           │
│  Racine/                                                                  │
│  ├── VIDEO_QUALITY_GUIDE.md        ← Guide COMPLET                        │
│  ├── QUALITY_IMPROVEMENTS_SUMMARY.md ← Résumé des changements            │
│  └── INTEGRATION_GUIDE.md          ← Guide technique                      │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ VÉRIFIER LA CONFIGURATION ──────────────────────────────────────────────┐
│                                                                           │
│  $ cd mapping/                                                            │
│  $ python quality_config.py                                              │
│                                                                           │
│  → Affiche la configuration actuelle                                      │
│  → Vérifie que JPEG_QUALITY et YOLO_DEVICE sont corrects                 │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ TESTER AVANT LANCEMENT ────────────────────────────────────────────────┐
│                                                                           │
│  $ cd mapping/                                                            │
│  $ python test_video_quality.py                                          │
│                                                                           │
│  Doit afficher:                                                           │
│  ✅ TEST 1: HighQualityRenderer                                           │
│  ✅ TEST 2: VehicleDataRenderer                                           │
│  ✅ TEST 3: VideoQualityEnhancer                                          │
│  ✅ TEST 4: JPEG Encoding                                                 │
│  ✅ TEST 5: Quality Configuration                                         │
│  ✅ ALL TESTS PASSED                                                      │
│                                                                           │
│  Si un test échoue → lire message d'erreur → ajuster quality_config.py  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ RÉSULTATS ATTENDUS ────────────────────────────────────────────────────┐
│                                                                           │
│  AVANT                          │ APRÈS                                   │
│  ────────────────────────────────┼─────────────────────────────────────  │
│  Vidéo floue, bruitée           │ Vidéo claire, débruitée               │
│  Texte petit, mal visible        │ Texte GRAND, LISIBLE, coloré         │
│  Données clignotantes            │ Données LISSES                        │
│  FPS: 5-8                        │ FPS: 12-15 ✅                         │
│  Configuration complexe          │ Configuration simple (presets) ✅     │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ DÉPANNAGE RAPIDE ──────────────────────────────────────────────────────┐
│                                                                           │
│  ❌ Erreur d'import:                                                      │
│     → Vérifie que t'es dans le dossier mapping/                          │
│     → python -c "import video_quality_enhancer"                          │
│                                                                           │
│  ❌ Vidéo toujours floue:                                                 │
│     → Vérifier VIDEO_DENOISE = True dans quality_config.py              │
│     → Essayer preset 'best_quality'                                      │
│                                                                           │
│  ❌ Texte pas visible:                                                    │
│     → Augmenter FONT_SIZE de 0.8 → 1.2                                  │
│     → Augmenter TEXT_THICKNESS de 2 → 3                                 │
│                                                                           │
│  ❌ FPS trop bas:                                                         │
│     → Essayer preset 'performance'                                       │
│     → Ou réduire YOLO_IMGSZ de 416 → 320                                │
│                                                                           │
│  ❌ GPU Memory Error:                                                     │
│     → Réduire YOLO_IMGSZ à 320                                           │
│     → Ou passer à YOLO_DEVICE = 'cpu'                                   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                           PRÊT À LANCER! 🚀                               ║
║                                                                            ║
║  $ cd mapping/                                                             ║
║  $ python launch_optimized.py                                              ║
║                                                                            ║
║  http://localhost:8050                                                    ║
║                                                                            ║
║  Les données de vitesse et distance devraient maintenant être:            ║
║  ✅ Claires et lisibles                                                    ║
║  ✅ Bien positionnées (pas de chevauchement)                              ║
║  ✅ Lisses (pas de cligotement)                                           ║
║  ✅ En couleurs distinctes (cyan/magenta)                                 ║
║  ✅ Avec IDs persistants (tracking)                                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# Aide supplémentaire
print("\n📖 Pour plus de détails:")
print("   • Lire: VIDEO_QUALITY_GUIDE.md")
print("   • Lire: QUALITY_IMPROVEMENTS_SUMMARY.md")
print("   • Éditer: mapping/quality_config.py")
print("\n✨ Bon dashboard! 🎉")
