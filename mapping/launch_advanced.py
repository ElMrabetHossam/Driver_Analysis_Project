"""
Quick Launch Script - ADVANCED ADAS MISSION CONTROL
Full Boeing/Tesla-Level Telemetry Dashboard
"""
import os
import sys

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ⚡  ADVANCED ADAS MISSION CONTROL SYSTEM  🛰️                ║
║        Professional Vehicle Telemetry Dashboard             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Starting advanced telemetry system...\n")

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Dependency checks
def check_requirements():
    missing = []
    try:
        import importlib.util as importlib_util
        def module_missing(name):
            return importlib_util.find_spec(name) is None
    except Exception:
        import pkgutil
        def module_missing(name):
            return pkgutil.find_loader(name) is None

    for pkg in ("dash", "dash_bootstrap_components", "plotly", "numpy", "pyproj", "flask", "cv2"):
        try:
            if module_missing(pkg):
                missing.append(pkg)
        except Exception:
            try:
                __import__(pkg)
            except Exception:
                missing.append(pkg)

    if missing:
        print("\n❌ Dépendances manquantes :", ", ".join(missing))
        print("   Installez-les avec :")
        print("     python -m pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ Toutes les dépendances Python sont installées.\n")

# Run dependency check
check_requirements()

# Import and run the advanced app
try:
    from app_advanced import app, load_and_process_data, create_layout
    
    # Load data
    load_and_process_data()
    
    # Create layout
    app.layout = create_layout()
    
    # Run server
    print("\n" + "="*60)
    print("🚀 ADVANCED ADAS MISSION CONTROL - LAUNCHING")
    print("="*60)
    print("\n📊 Features Enabled:")
    print("   ✅ Ground Truth Path Projection (ECEF → Camera)")
    print("   ✅ Multi-Target Radar Display with Fusion")
    print("   ✅ Artificial Horizon (Pitch, Roll, Yaw)")
    print("   ✅ 4-Wheel Dynamics with Slip Detection")
    print("   ✅ Advanced Metrics (Centrifugal Force, Yaw Rate, etc.)")
    print("   ✅ Understeering/Oversteering Detection")
    print("   ✅ YOLO Object Detection + Tracking")
    print("\n🌐 Open your browser and navigate to:")
    print("   👉 http://127.0.0.1:8051/")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=False, port=8051, host='127.0.0.1')
    
except KeyboardInterrupt:
    print("\n\n✋ Server stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n Please make sure all dependencies are installed:")
    print("   python -m pip install -r requirements.txt")
    sys.exit(1)
