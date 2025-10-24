import os
import shutil
import datetime

def create_mac_app_bundle():
    """Create a native macOS .app bundle that works with double-click"""
    
    print("Creating macOS Application Bundle...")
    
    app_name = "EmotionDetector"
    app_bundle = f"{app_name}.app"
    
    # Remove existing app if it exists
    if os.path.exists(app_bundle):
        shutil.rmtree(app_bundle)
    
    # Create .app bundle structure
    contents_dir = f"{app_bundle}/Contents"
    macos_dir = f"{contents_dir}/MacOS"
    resources_dir = f"{contents_dir}/Resources"
    
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)
    
    print(f"Created {app_bundle} structure")
    
    # Copy emotion_detector.py to Resources
    if os.path.exists('emotion_detector.py'):
        shutil.copy2('emotion_detector.py', resources_dir)
        print("Copied emotion_detector.py")
    else:
        print("emotion_detector.py not found!")
        return False
    
    # Create Info.plist
    info_plist = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>EmotionDetector</string>
    <key>CFBundleIdentifier</key>
    <string>com.ardasengec.emotiondetector</string>
    <key>CFBundleName</key>
    <string>Emotion Detector</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>NSCameraUsageDescription</key>
    <string>This application needs camera access to detect emotions during exam monitoring.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>CFBundleDocumentTypes</key>
    <array/>
</dict>
</plist>'''
    
    with open(f"{contents_dir}/Info.plist", 'w', encoding='utf-8') as f:
        f.write(info_plist)
    print("Created Info.plist")
    
    # Create the main executable script
    executable_script = '''#!/bin/bash

# EmotionDetector.app Main Executable
# This script sets up Python environment and runs the application

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="$APP_DIR/Resources"

# Change to resources directory
cd "$RESOURCES_DIR"

# Function to show notification
show_notification() {
    osascript -e "display notification \\"$1\\" with title \\"Emotion Detector\\""
}

# Function to show dialog
show_dialog() {
    osascript -e "display dialog \\"$1\\" with title \\"Emotion Detector\\" buttons {\\"OK\\"} default button \\"OK\\""
}

# Function to show progress
show_progress() {
    osascript -e "display dialog \\"$1\\" with title \\"Emotion Detector\\" giving up after 3"
}

# Check if Python3 is installed
if ! command -v python3 >/dev/null 2>&1; then
    show_dialog "Python 3 is not installed. Please install Python 3 from python.org and try again."
    exit 1
fi

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"
if [ $? -ne 0 ]; then
    show_dialog "Python 3.7+ is required. Please update Python and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_DIR="$RESOURCES_DIR/.emotion_env"
if [ ! -d "$VENV_DIR" ]; then
    show_progress "Setting up environment for first time..."
    python3 -m venv "$VENV_DIR"
    
    # Activate and install packages
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    python -m pip install --upgrade pip --quiet
    
    # Install packages with Apple Silicon optimization
    if [[ $(uname -m) == 'arm64' ]]; then
        # Apple Silicon (M1/M2) optimized installation
        python -m pip install "numpy>=1.21.0,<2.0.0" --quiet
        python -m pip install "opencv-python>=4.5.0" --quiet
        python -m pip install "pandas>=1.3.0,<2.1.0" --quiet
        python -m pip install "Pillow>=8.0.0" --quiet
        python -m pip install "scikit-learn>=1.1.0" --quiet
        python -m pip install "tensorflow-macos>=2.9.0" --quiet
        python -m pip install "tensorflow-metal" --quiet 2>/dev/null || true
        python -m pip install "deepface>=0.0.79" --quiet
    else
        # Intel Mac installation
        python -m pip install "numpy>=1.21.0,<2.0.0" --quiet
        python -m pip install "opencv-python>=4.5.0" --quiet
        python -m pip install "pandas>=1.3.0,<2.1.0" --quiet
        python -m pip install "Pillow>=8.0.0" --quiet
        python -m pip install "scikit-learn>=1.1.0" --quiet
        python -m pip install "tensorflow>=2.8.0,<2.16.0" --quiet
        python -m pip install "deepface>=0.0.79" --quiet
    fi
    
    show_notification "Environment setup complete!"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if all packages are available
python -c "
import sys
missing = []
try:
    import cv2
except ImportError:
    missing.append('opencv-python')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    import tensorflow
except ImportError:
    missing.append('tensorflow')
try:
    from deepface import DeepFace
except ImportError:
    missing.append('deepface')

if missing:
    print('Missing packages:', missing)
    sys.exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    show_dialog "Some packages are missing. The application will try to reinstall them."
    
    # Reinstall missing packages
    source "$VENV_DIR/bin/activate"
    pip install --upgrade opencv-python numpy tensorflow deepface pandas Pillow scikit-learn --quiet
fi

# Run the application
show_notification "Starting Emotion Detector..."

# Run in background and capture output
python emotion_detector.py 2>&1 | while read line; do
    echo "$line"
done

# Check exit status
if [ $? -eq 0 ]; then
    show_notification "Emotion Detector finished successfully"
else
    show_dialog "Emotion Detector encountered an error. Check the terminal for details."
fi
'''
    
    # Write executable script
    executable_path = f"{macos_dir}/EmotionDetector"
    with open(executable_path, 'w', encoding='utf-8') as f:
        f.write(executable_script)
    
    # Make executable
    os.chmod(executable_path, 0o755)
    print("Created main executable")
    
    # Create a simple launcher script for debugging
    debug_script = '''#!/bin/bash
echo "Emotion Detector Debug Mode"
echo "=========================="

# Get app resources directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$SCRIPT_DIR/Resources"

echo "Resources directory: $RESOURCES_DIR"
echo "Contents:"
ls -la "$RESOURCES_DIR"

echo ""
echo "Python version:"
python3 --version

echo ""
echo "Starting application in debug mode..."
cd "$RESOURCES_DIR"

# Check if virtual environment exists
if [ -d ".emotion_env" ]; then
    echo "Activating virtual environment..."
    source .emotion_env/bin/activate
    echo "Python path: $(which python)"
    echo "Installed packages:"
    pip list | grep -E "(opencv|numpy|tensorflow|deepface)"
fi

echo ""
echo "Running emotion_detector.py..."
python3 emotion_detector.py

echo ""
echo "Press Enter to close..."
read
'''
    
    with open(f"{contents_dir}/debug.sh", 'w', encoding='utf-8') as f:
        f.write(debug_script)
    os.chmod(f"{contents_dir}/debug.sh", 0o755)
    print("Created debug script")
    
    # Create README for Mac usage (without Unicode arrows)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme = f'''# Emotion Detector for macOS

## Installation:
1. Copy EmotionDetector.app to your Applications folder
2. Double-click to run

## First Run:
- The app will automatically set up Python environment
- This may take 5-10 minutes on first launch
- You will see progress notifications

## Permissions:
- Allow camera access when prompted
- The app needs camera permission to work

## Troubleshooting:

### If the app won't open:
1. Right-click > Open (to bypass Gatekeeper)
2. Or run in Terminal: open EmotionDetector.app

### Debug mode:
If you have issues, run the debug script:
cd EmotionDetector.app/Contents
./debug.sh

### Manual installation:
If automatic setup fails:
cd EmotionDetector.app/Contents/Resources
python3 -m pip install opencv-python numpy tensorflow deepface pandas Pillow scikit-learn
python3 emotion_detector.py

## Requirements:
- macOS 10.13+
- Python 3.7+
- Camera access
- Internet connection (for first setup)

## Created: {current_time}
'''
    
    with open(f"{contents_dir}/README.md", 'w', encoding='utf-8') as f:
        f.write(readme)
    print("Created README")
    
    # Create run instructions text file
    instructions = '''EMOTION DETECTOR - MAC INSTRUCTIONS

Quick Start:
1. Double-click EmotionDetector.app
2. Wait for first-time setup (5-10 minutes)
3. Allow camera access when prompted
4. Application will start automatically

If app doesn't start:
1. Right-click EmotionDetector.app
2. Select "Open" from menu
3. Click "Open" in security dialog

For debugging:
1. Open Terminal
2. Navigate to EmotionDetector.app/Contents
3. Run: ./debug.sh

Requirements:
- Python 3.7+ (will be set up automatically)
- Internet connection for first run
- Camera permission

The app will create its own Python environment and install all needed packages automatically.
'''
    
    with open(f"{contents_dir}/HOW_TO_RUN.txt", 'w', encoding='utf-8') as f:
        f.write(instructions)
    print("Created instructions file")
    
    print(f"\nmacOS Application Bundle created successfully!")
    print(f"Location: {os.path.abspath(app_bundle)}")
    print(f"Double-click '{app_bundle}' to run")
    print(f"Copy to Applications folder for easy access")
    print(f"Debug mode: run '{app_bundle}/Contents/debug.sh' if needed")
    
    return True

def create_dmg_installer():
    """Create a DMG installer for easy distribution"""
    
    app_name = "EmotionDetector"
    dmg_name = f"{app_name}_Installer"
    
    print(f"\nCreating DMG installer: {dmg_name}")
    
    try:
        # Create temporary directory for DMG contents
        dmg_contents = f"{dmg_name}_contents"
        if os.path.exists(dmg_contents):
            shutil.rmtree(dmg_contents)
        os.makedirs(dmg_contents)
        
        # Copy app bundle to DMG contents
        if os.path.exists(f"{app_name}.app"):
            shutil.copytree(f"{app_name}.app", f"{dmg_contents}/{app_name}.app")
            print(f"Copied {app_name}.app to DMG contents")
        
        # Create install instructions
        install_txt = """Emotion Detector Installation

1. Drag EmotionDetector.app to Applications folder
2. Double-click EmotionDetector.app to run
3. Allow camera access when prompted
4. First run may take 5-10 minutes to set up

For support, see README in the app bundle.
"""
        with open(f"{dmg_contents}/Install Instructions.txt", 'w', encoding='utf-8') as f:
            f.write(install_txt)
        
        print(f"Created DMG contents directory: {dmg_contents}")
        print(f"Ready for DMG creation")
        print(f"To create DMG on Mac, run:")
        print(f"  hdiutil create -volname '{app_name}' -srcfolder '{dmg_contents}' -ov -format UDZO '{dmg_name}.dmg'")
        
        return True
        
    except Exception as e:
        print(f"DMG creation setup failed: {e}")
        return False

def create_simple_mac_script():
    """Create a simple script for Mac users who just want to run it easily"""
    
    simple_script = '''#!/bin/bash

# Simple Emotion Detector Runner for Mac
# Just double-click this file to run the app

echo "Emotion Detector - Simple Runner"
echo "================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for emotion_detector.py
if [ ! -f "emotion_detector.py" ]; then
    echo "ERROR: emotion_detector.py not found in current directory"
    echo "Make sure this script is in the same folder as emotion_detector.py"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 not found"
    echo "Please install Python 3 from https://python.org"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Python found: $(python3 --version)"

# Create virtual environment if needed
if [ ! -d "emotion_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv emotion_env
fi

# Activate environment
echo "Activating environment..."
source emotion_env/bin/activate

# Install packages
echo "Installing packages (this may take a few minutes)..."
pip install --upgrade pip --quiet

# Install with NumPy compatibility
pip install "numpy<2.0" --quiet
pip install opencv-python pandas Pillow scikit-learn --quiet

# TensorFlow for Mac
if [[ $(uname -m) == 'arm64' ]]; then
    echo "Installing TensorFlow for Apple Silicon..."
    pip install tensorflow-macos tensorflow-metal --quiet
else
    echo "Installing TensorFlow for Intel Mac..."
    pip install tensorflow --quiet
fi

# DeepFace
pip install deepface --quiet

echo ""
echo "Starting Emotion Detector..."
python emotion_detector.py

echo ""
echo "Application finished."
read -p "Press Enter to close..."
'''
    
    with open("run_emotion_detector_mac.sh", 'w', encoding='utf-8') as f:
        f.write(simple_script)
    
    os.chmod("run_emotion_detector_mac.sh", 0o755)
    print("Created simple Mac runner: run_emotion_detector_mac.sh")

if __name__ == "__main__":
    print("=== macOS App Bundle Creator ===")
    print("1. Create .app bundle (recommended)")
    print("2. Create .app + DMG installer")
    print("3. Create simple script runner")
    print()
    
    choice = input("Choose option (1, 2, 3, or Enter for .app only): ").strip()
    
    success = False
    
    if choice == "3":
        create_simple_mac_script()
        success = True
    else:
        success = create_mac_app_bundle()
        if success and choice == "2":
            create_dmg_installer()
    
    if success:
        print("\nUsage Instructions:")
        print("1. Copy the created files to your Mac")
        if choice == "3":
            print("2. Double-click 'run_emotion_detector_mac.sh'")
        else:
            print("2. Double-click 'EmotionDetector.app'")
        print("3. First run will take 5-10 minutes to set up")
        print("4. Allow camera access when prompted")
    else:
        print("Failed to create macOS files")