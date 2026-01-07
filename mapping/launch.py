"""
Quick Launch Script
Runs the Vehicle Tracking Visualization Dashboard
"""
import os
import sys

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚗  VEHICLE TRACKING VISUALIZATION SYSTEM  🗺️               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Starting system...\n")

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# New: dependency checks to provide clear instructions if something manque
def check_requirements():
    # Robust module existence check: prefer importlib.util.find_spec, fallback to pkgutil.find_loader
    missing = []
    try:
        import importlib.util as importlib_util  # may fail if importlib is shadowed
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
            # last-resort: try importing directly
            try:
                __import__(pkg)
            except Exception:
                missing.append(pkg)

    # FFmpeg is optional but warn if missing
    ffmpeg_missing = False
    try:
        from shutil import which
        if which("ffmpeg") is None:
            ffmpeg_missing = True
    except Exception:
        ffmpeg_missing = True

    if missing:
        print("\n❌ Dépendances manquantes :", ", ".join(missing))
        print("   Installez-les avec :")
        print("     python -m pip install -r requirements.txt")
        print("   Ou installez manuellement, par exemple :")
        print("     python -m pip install " + " ".join(missing))
        sys.exit(1)

    if ffmpeg_missing:
        print("\n⚠️  FFmpeg non trouvé dans le PATH. L'extraction automatique des frames échouera.")
        print("   Installez FFmpeg: https://ffmpeg.org/download.html")
    else:
        print("✅ Dépendances Python et FFmpeg détectées.")

def check_ffmpeg():
    """
    Check if FFmpeg is available, and provide a fallback mechanism.
    """
    from shutil import which
    ffmpeg_path = os.getenv("FFMPEG_PATH")  # Allow custom FFmpeg path via environment variable
    if ffmpeg_path and os.path.isfile(ffmpeg_path):
        print(f"✅ FFmpeg found at custom path: {ffmpeg_path}")
        return ffmpeg_path
    elif which("ffmpeg") is not None:
        print("✅ FFmpeg found in PATH.")
        return "ffmpeg"
    else:
        print("\n⚠️  FFmpeg not found in PATH or custom path.")
        print("   Install FFmpeg: https://ffmpeg.org/download.html")
        print("   Or set the FFMPEG_PATH environment variable to its location.")
        return None

# Run dependency check before importing the heavy app
check_requirements()
ffmpeg_path = check_ffmpeg()

# Import and run the app
try:
    from app import app, load_and_process_data, create_layout

    # Load data with enhanced error handling
    print("Loading vehicle tracking data...")
    try:
        load_and_process_data()
        print("✓ Successfully loaded vehicle tracking data.")
    except FileNotFoundError as e:
        print(f"⚠️ Data file not found: {e}")
        print("   Please ensure all required data files are in the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error while loading data: {e}")
        sys.exit(1)

    # Create layout with debugging logs
    try:
        print("Creating dashboard layout...")
        app.layout = create_layout()
        print("✓ Layout created successfully.")
    except Exception as e:
        print(f"❌ Error creating layout: {e}")
        sys.exit(1)

    # Run server
    print("\n" + "="*60)
    print("🚀 Dashboard is launching...")
    print("="*60)
    print("\n📊 Open your browser and navigate to:")
    print("   👉 http://127.0.0.1:8050/")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)  # Set debug=True for development

except KeyboardInterrupt:
    print("\n\n✋ Server stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease make sure all dependencies are installed:")
    print("   python -m pip install -r requirements.txt")
    sys.exit(1)
    # Example: Wrapping a callback function
    # @app.callback(Output("output-id", "children"), [Input("input-id", "value")])
    # @safe_callback
    # def example_callback(input_value):
    #     # ...existing callback logic...
    #     return f"Processed value: {input_value}"

    # Validate layout and component IDs
    try:
        print("Validating layout and component IDs...")
        app.layout = create_layout()
        logging.debug("Layout successfully created.")
    except Exception as e:
        logging.error(f"Error in layout creation: {e}")
        sys.exit(1)

    # Run server
    print("\n" + "="*60)
    print("🚀 Dashboard is launching...")
    print("="*60)
    print("\n📊 Open your browser and navigate to:")
    print("   👉 http://127.0.0.1:8050/")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=False, port=8050)

except KeyboardInterrupt:
    print("\n\n✋ Server stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease make sure all dependencies are installed:")
    print("   python -m pip install -r requirements.txt")
    sys.exit(1)
