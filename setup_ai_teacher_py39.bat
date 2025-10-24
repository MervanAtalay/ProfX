REM filepath: c:\Users\320200432\Desktop\PY-EMR\setup_ai_teacher_py39.bat
@echo off
echo 🤖 AI Teacher Python 3.9 Virtual Environment Setup
echo ==================================================

cd /d c:\Users\320200432\Desktop\PY-EMR\

echo 🗑️ Removing old virtual environment...
if exist ai_teacher_env rmdir /s /q ai_teacher_env

echo 🐍 Creating new Python 3.9 virtual environment...
python -m venv ai_teacher_env

echo 🔌 Activating virtual environment...
call ai_teacher_env\Scripts\activate

echo 🧪 Testing Python version...
python --version
pip --version

echo 📦 Upgrading pip...
python -m pip install --upgrade pip

echo 📥 Installing AI Teacher requirements...
pip install pyaudio
pip install SpeechRecognition
pip install pyttsx3
pip install opencv-python
pip install deepface
pip install openai-whisper
pip install tensorflow
pip install numpy
pip install pandas

echo 🎤 Testing microphone...
python soundcheck.py

echo ✅ Setup complete! 
echo 💡 To activate environment: ai_teacher_env\Scripts\activate
echo 💡 To run AI Teacher: python AI_teach.py
pause