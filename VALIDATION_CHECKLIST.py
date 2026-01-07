"""
VALIDATION CHECKLIST - Vérifier que tout fonctionne correctement
Run: python VALIDATION_CHECKLIST.py
"""

import os
import sys

def print_header(text):
    print("\n" + "█"*70)
    print(f"█ {text.center(68)} █")
    print("█"*70)

def print_section(text):
    print(f"\n┌─ {text} " + "─"*(65-len(text)) + "┐")

def check_file(filepath):
    """Vérifier qu'un fichier existe"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"   {status} {os.path.basename(filepath)}")
    return exists

def check_import(module_name):
    """Vérifier qu'un module peut être importé"""
    try:
        __import__(module_name)
        print(f"   ✅ {module_name}")
        return True
    except ImportError as e:
        print(f"   ❌ {module_name}: {e}")
        return False

def main():
    print_header("VIDEO QUALITY ENHANCEMENT - VALIDATION CHECKLIST")
    
    # 1. Fichiers créés
    print_section("1. FICHIERS CRÉÉS")
    
    files_to_check = [
        "mapping/video_quality_enhancer.py",
        "mapping/quality_config.py",
        "mapping/test_video_quality.py",
        "mapping/launch_optimized.py",
        "VIDEO_QUALITY_GUIDE.md",
        "QUALITY_IMPROVEMENTS_SUMMARY.md",
        "QUICK_START.py",
        "INTEGRATION_GUIDE.md",
    ]
    
    all_files_exist = all(check_file(f) for f in files_to_check)
    
    # 2. Fichiers modifiés
    print_section("2. FICHIERS MODIFIÉS")
    
    modified_files = [
        "mapping/app.py",
        "mapping/video_processor.py",
        "mapping/data_loader.py",
    ]
    
    all_modified = all(check_file(f) for f in modified_files)
    
    # 3. Modules requérants
    print_section("3. DÉPENDANCES")
    
    dependencies = [
        "cv2",       # OpenCV
        "numpy",     # NumPy
        "scipy",     # SciPy (pour vehicle_tracker)
        "dash",      # Dash
        "plotly",    # Plotly
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - MANQUANT!")
            missing_deps.append(dep)
    
    # 4. Configuration
    print_section("4. CONFIGURATION")
    
    try:
        os.chdir("mapping")
        sys.path.insert(0, os.getcwd())
        import quality_config
        
        print(f"   ✅ quality_config importé")
        print(f"   • JPEG_QUALITY: {quality_config.JPEG_QUALITY}")
        print(f"   • VIDEO_DENOISE: {quality_config.VIDEO_DENOISE}")
        print(f"   • YOLO_DEVICE: {quality_config.YOLO_DEVICE}")
        print(f"   • YOLO_IMGSZ: {quality_config.YOLO_IMGSZ}")
        
        os.chdir("..")
    except Exception as e:
        print(f"   ❌ Erreur configuration: {e}")
    
    # 5. Modules Python
    print_section("5. MODULES PYTHON")
    
    modules_to_check = [
        "video_quality_enhancer",
        "quality_config",
        "vehicle_tracker",
        "enhanced_overlay",
        "smoothing_filter",
    ]
    
    os.chdir("mapping")
    sys.path.insert(0, os.getcwd())
    
    all_imports_ok = True
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            all_imports_ok = False
    
    os.chdir("..")
    
    # 6. Résumé
    print_section("RÉSUMÉ")
    
    all_good = all_files_exist and all_modified and all_imports_ok and not missing_deps
    
    print("\n   Fichiers créés:     ", "✅ OK" if all_files_exist else "❌ PROBLÈME")
    print("   Fichiers modifiés:  ", "✅ OK" if all_modified else "❌ PROBLÈME")
    print("   Dépendances:        ", "✅ OK" if not missing_deps else "❌ MANQUANTES")
    print("   Modules Python:     ", "✅ OK" if all_imports_ok else "❌ ERREUR")
    
    if missing_deps:
        print(f"\n   ⚠️  Dépendances manquantes: {', '.join(missing_deps)}")
        print("   Installation: pip install -r requirements.txt")
    
    print("\n" + "█"*70)
    
    if all_good:
        print("█" + " "*68 + "█")
        print("█" + "✅ ALL CHECKS PASSED - READY TO LAUNCH!".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)
        
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║  Prochaines étapes:                                                       ║
║                                                                            ║
║  1. Tester les modules:                                                    ║
║     cd mapping/                                                            ║
║     python test_video_quality.py                                           ║
║                                                                            ║
║  2. Lancer le dashboard:                                                   ║
║     python launch_optimized.py                                             ║
║                                                                            ║
║  3. Accéder à: http://localhost:8050                                       ║
║                                                                            ║
║  4. Vérifier:                                                              ║
║     ✓ Vidéo claire (pas floue)                                             ║
║     ✓ Texte vitesse/distance visible                                       ║
║     ✓ Pas de cligotement                                                   ║
║     ✓ FPS acceptable (> 10)                                                ║
║                                                                            ║
║  5. Si besoin d'ajustements:                                               ║
║     Éditer mapping/quality_config.py                                       ║
║     Consulter VIDEO_QUALITY_GUIDE.md                                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
        """)
        return 0
    else:
        print("█" + " "*68 + "█")
        print("█" + "❌ SOME CHECKS FAILED - SEE ABOVE FOR DETAILS".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)
        
        print("\n⚠️  Corrections nécessaires:")
        if not all_files_exist:
            print("   • Vérifier que tous les fichiers sont créés")
        if not all_modified:
            print("   • Vérifier que app.py, video_processor.py sont modifiés")
        if missing_deps:
            print(f"   • Installer: pip install {' '.join(missing_deps)}")
        if not all_imports_ok:
            print("   • Vérifier les imports - voir détails ci-dessus")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
