import os
import sys
import subprocess
import shutil
import platform
import time
import datetime

def safe_remove_directory(path, max_attempts=3):
    """Safely remove directory with retry mechanism"""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(path):
                time.sleep(1)
                shutil.rmtree(path, ignore_errors=True)
                
                if not os.path.exists(path):
                    return True
                    
        except PermissionError as e:
            print(f"Warning: Attempt {attempt + 1}: Permission error - {e}")
            if attempt < max_attempts - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("Could not remove directory. Creating with different name...")
                return False
        except Exception as e:
            print(f"Warning: Unexpected error: {e}")
            return False
    
    return False

def get_python_version():
    """Get current Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"

def create_compatible_requirements():
    """Create requirements.txt compatible with current Python version"""
    python_version = get_python_version()
    print(f"Python version: {python_version}")
    
    # TensorFlow compatibility matrix
    if python_version in ["3.7", "3.8"]:
        tf_version = "tensorflow>=2.6.0,<2.12.0"
    elif python_version == "3.9":
        tf_version = "tensorflow>=2.8.0,<2.16.0"
    elif python_version in ["3.10", "3.11"]:
        tf_version = "tensorflow>=2.10.0"
    else:
        tf_version = "tensorflow>=2.6.0"  # fallback
    
    requirements = f"""deepface>=0.0.79
opencv-python>=4.5.0
{tf_version}
numpy>=1.19.0
pandas>=1.3.0
Pillow>=8.0.0
scikit-learn>=1.0.0
protobuf>=3.19.0,<4.0.0"""
    
    return requirements

def create_portable_version():
    """Create a portable Python version that works on any platform"""
    print("Creating portable cross-platform version...")
    
    # Create portable directory with timestamp to avoid conflicts
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = "EmotionDetector_Portable"
    portable_dir = base_name
    
    # If directory exists, try to remove it safely
    if os.path.exists(portable_dir):
        print(f"Directory {portable_dir} exists, attempting to remove...")
        if not safe_remove_directory(portable_dir):
            # If removal fails, use timestamped name
            portable_dir = f"{base_name}_{timestamp}"
            print(f"Using alternative name: {portable_dir}")
    
    # Create directory
    try:
        os.makedirs(portable_dir, exist_ok=True)
        print(f"Created directory: {portable_dir}")
    except Exception as e:
        print(f"Failed to create directory: {e}")
        return False
    
    # Check if main script exists
    if not os.path.exists('emotion_detector.py'):
        print("emotion_detector.py not found in current directory")
        return False
    
    # Copy main script
    try:
        shutil.copy2('emotion_detector.py', portable_dir)
        print("Copied emotion_detector.py")
    except Exception as e:
        print(f"Failed to copy main script: {e}")
        return False
    
    # Create compatible requirements.txt
    requirements = create_compatible_requirements()
    
    try:
        with open(f"{portable_dir}/requirements.txt", 'w') as f:
            f.write(requirements)
        print("Created requirements.txt")
    except Exception as e:
        print(f"Failed to create requirements.txt: {e}")
        return False
    
    # Create Windows batch file
    windows_script = """@echo off
title Emotion Detector Setup
echo ========================================
echo    Emotion Detector Setup for Windows
echo ========================================

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing packages...
echo This may take several minutes...

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install packages with fallback options
echo Installing OpenCV...
python -m pip install opencv-python>=4.5.0 || (
    echo Trying alternative OpenCV...
    python -m pip install opencv-python-headless>=4.5.0
)

echo Installing core packages...
python -m pip install numpy>=1.19.0 pandas>=1.3.0 Pillow>=8.0.0

echo Installing TensorFlow...
python -m pip install tensorflow>=2.6.0 || (
    echo Trying TensorFlow CPU version...
    python -m pip install tensorflow-cpu>=2.6.0
)

echo Installing DeepFace...
python -m pip install deepface>=0.0.79 scikit-learn>=1.0.0

echo.
echo Installation complete! Starting application...
echo.

python emotion_detector.py

echo.
echo Application closed. Press any key to exit.
pause
"""
    
    try:
        with open(f"{portable_dir}/RUN_WINDOWS.bat", 'w') as f:
            f.write(windows_script)
        print("Created RUN_WINDOWS.bat")
    except Exception as e:
        print(f"Failed to create Windows script: {e}")
    
    # Create macOS/Linux shell script - FIXED VERSION
    mac_linux_script = '''#!/bin/bash

echo "========================================"
echo "   Emotion Detector Setup for Mac/Linux"
echo "========================================"

# Check Python installation
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found"
    echo "Please install Python 3.7+ first"
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "Python3 found. Checking version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3.7+ required"
    python3 --version
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "Python version OK"

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install packages step by step
echo "Installing packages (this may take several minutes)..."

echo "Installing OpenCV..."
python3 -m pip install "opencv-python>=4.5.0"
if [ $? -ne 0 ]; then
    echo "Trying alternative OpenCV..."
    python3 -m pip install "opencv-python-headless>=4.5.0"
fi

echo "Installing core packages..."
python3 -m pip install "numpy>=1.19.0" "pandas>=1.3.0" "Pillow>=8.0.0"

echo "Installing TensorFlow..."
python3 -m pip install "tensorflow>=2.6.0"
if [ $? -ne 0 ]; then
    echo "Trying TensorFlow CPU version..."
    python3 -m pip install "tensorflow-cpu>=2.6.0"
fi

echo "Installing DeepFace..."
python3 -m pip install "deepface>=0.0.79" "scikit-learn>=1.0.0"

echo ""
echo "Installation complete! Starting application..."
echo ""

python3 emotion_detector.py

echo ""
echo "Application closed. Press Enter to exit..."
read
'''
    
    try:
        with open(f"{portable_dir}/run_mac_linux.sh", 'w', newline='\n') as f:
            f.write(mac_linux_script)
        print("Created run_mac_linux.sh")
        
        # Make shell script executable on Unix systems
        try:
            os.chmod(f"{portable_dir}/run_mac_linux.sh", 0o755)
            print("Made shell script executable")
        except:
            pass  # Windows doesn't support chmod
            
    except Exception as e:
        print(f"Failed to create Mac/Linux script: {e}")
    
    # Create simple Python installer (without emojis)
    simple_installer = """#!/usr/bin/env python3
# Simple installer for Emotion Detector
import subprocess
import sys
import os

def main():
    print("Emotion Detector Installer")
    print("Python version:", sys.version)
    
    # Install packages
    packages = [
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0", 
        "tensorflow>=2.6.0",
        "deepface>=0.0.79",
        "Pillow>=8.0.0",
        "scikit-learn>=1.0.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        print("Installing", package, "...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print("SUCCESS:", package, "installed")
        except:
            print("FAILED:", package)
            failed_packages.append(package)
    
    if failed_packages:
        print("\\nFailed packages:", failed_packages)
        print("Try installing manually with: pip install [package_name]")
    else:
        print("\\nAll packages installed successfully!")
    
    # Try to run the application
    print("\\nStarting Emotion Detector...")
    try:
        exec(open('emotion_detector.py').read())
        print("Application started successfully!")
    except Exception as e:
        print("Failed to start:", str(e))
    
    input("\\nPress Enter to exit...")

if __name__ == "__main__":
    main()
"""
    
    try:
        with open(f"{portable_dir}/INSTALL_AND_RUN.py", 'w', encoding='utf-8') as f:
            f.write(simple_installer)
        print("Created INSTALL_AND_RUN.py")
    except Exception as e:
        print(f"Failed to create Python installer: {e}")
    
    # Create README
    readme = f"""# Emotion Detector - Portable Version

Created: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Python Version: {get_python_version()}

## Quick Start

### Windows:
1. Double-click `RUN_WINDOWS.bat`
2. Wait for installation (may take 5-10 minutes)
3. Application will start automatically

### Mac/Linux:
1. Open Terminal in this folder
2. Run: `chmod +x run_mac_linux.sh`
3. Run: `./run_mac_linux.sh`
4. Or run: `bash run_mac_linux.sh`

### Alternative (All Platforms):
1. Double-click `INSTALL_AND_RUN.py`
2. Follow the prompts

## Manual Installation:
1. Open command prompt/terminal in this folder
2. Run: `pip install -r requirements.txt`
3. Run: `python emotion_detector.py`

## Requirements:
- Python 3.7+ (detected: {get_python_version()})
- Internet connection for installation
- Webcam access
- 4GB+ RAM recommended

## Troubleshooting:

### Mac Terminal Commands:
```bash
# Navigate to folder
cd /path/to/EmotionDetector_Portable

# Make executable
chmod +x run_mac_linux.sh

# Run script
./run_mac_linux.sh

# Alternative
bash run_mac_linux.sh
```

### Common Issues:
- If installation fails, try updating Python
- For camera issues, check system privacy settings
- Run as administrator/sudo if needed
- On Mac: Allow camera access in System Preferences

## Files:
- `emotion_detector.py` - Main application
- `requirements.txt` - Required packages
- `RUN_WINDOWS.bat` - Windows installer
- `run_mac_linux.sh` - Mac/Linux installer
- `INSTALL_AND_RUN.py` - Cross-platform installer
- `README.md` - This file

## Platform-Specific Notes:

### Windows:
- Make sure Python is in PATH
- Install from Microsoft Store or python.org
- May need to run as administrator

### macOS:
- Use Homebrew: `brew install python`
- Allow camera access in System Preferences
- May need to install Xcode command line tools

### Linux:
- Install python3-dev: `sudo apt install python3-dev`
- May need additional packages for OpenCV
"""
    
    try:
        with open(f"{portable_dir}/README.md", 'w', encoding='utf-8') as f:
            f.write(readme)
        print("Created README.md")
    except Exception as e:
        print(f"Failed to create README: {e}")
    
    print(f"\\nPortable version created successfully!")
    print(f"Location: {os.path.abspath(portable_dir)}")
    print(f"Windows: Double-click 'RUN_WINDOWS.bat'")
    print(f"Mac/Linux: Run './run_mac_linux.sh' in terminal")
    print(f"This folder can be copied to any computer")
    
    return True

def create_windows_exe():
    """Create Windows .exe file"""
    print("Building Windows executable...")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], check=True)
        except:
            print("Failed to install PyInstaller")
            return False
    
    # Clean up
    safe_remove_directory('build')
    safe_remove_directory('dist')
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        '--noconsole',
        '--name', 'EmotionDetector',
        'emotion_detector.py'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Windows .exe created successfully!")
        print("Location: dist/EmotionDetector.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Cross-Platform Build Tool ===")
    print(f"Python version: {get_python_version()}")
    print("1. Windows executable (.exe)")
    print("2. Portable version (recommended)")
    print()
    
    choice = input("Choose option (1 or 2, or Enter for portable): ").strip()
    
    if choice == "1":
        create_windows_exe()
    else:
        create_portable_version()