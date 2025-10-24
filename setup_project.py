

"""
Setup script for the Real-time Emotion Detection project.
This script helps set up the virtual environment and install dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🎯 Setting up Real-time Emotion Detection Project")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print("❌ Python 3.9 or newer is required!")
        print(f"Current version: {python_version.major}.{python_version.minor}")
        return
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    
    # Create virtual environment
    if not run_command("python -m venv emotion_detection_env", "Creating virtual environment"):
        return
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "emotion_detection_env\\Scripts\\activate"
        pip_command = "emotion_detection_env\\Scripts\\pip"
    else:  # Linux/Mac
        activate_script = "source emotion_detection_env/bin/activate"
        pip_command = "emotion_detection_env/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_command} install --upgrade pip", "Upgrading pip"):
        return
    
    if not run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies"):
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   emotion_detection_env\\Scripts\\activate")
    else:
        print("   source emotion_detection_env/bin/activate")
    print("2. Run the emotion detector:")
    print("   python emotion_detector.py")
    print("\n💡 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'c' to switch camera")

if __name__ == "__main__":
    main()