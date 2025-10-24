@echo off
echo 🐍 Python 3.9 Setup for AI Teacher
echo ================================

echo 📥 Downloading Python 3.9.13...
curl -o python-3.9.13-amd64.exe https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe

echo 🔧 Installing Python 3.9.13...
python-3.9.13-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

echo ⏳ Waiting for installation...
timeout /t 30

echo 🧪 Testing Python installation...
python --version

echo 📦 Installing AI Teacher requirements...
python -m pip install --upgrade pip
pip install pyaudio SpeechRecognition pyttsx3 opencv-python deepface

echo ✅ Setup complete!
pause