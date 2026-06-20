import sys
print(f"Python: {sys.version}")

packages = {
    "streamlit": "streamlit",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "moviepy": "moviepy",
    "edge-tts": "edge_tts",
    "Pillow": "PIL",
    "mutagen": "mutagen",
    "imageio": "imageio",
    "imageio-ffmpeg": "imageio_ffmpeg",
}

all_ok = True
for pkg_name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {pkg_name} ({version})")
    except ImportError as e:
        print(f"  [MISSING] {pkg_name} — {e}")
        all_ok = False

print()
if all_ok:
    print("✅ All packages are installed. Your app is ready to run!")
    print("   Run it with:  streamlit run app.py")
else:
    print("❌ Some packages are missing. Install them with:")
    print("   pip install -r requirements.txt")
