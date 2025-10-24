@echo off
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing TensorFlow...
pip install tensorflow --no-cache-dir

echo.
echo Installing DeepFace...
pip install deepface --no-cache-dir

echo.
echo Installing FastAPI and web tools...
pip install fastapi uvicorn[standard] python-multipart websockets --no-cache-dir

echo.
echo Installing image processing...
pip install opencv-python numpy pillow --no-cache-dir

echo.
echo Installing Gemini API...
pip install google-genai --no-cache-dir

echo.
echo Installing pyttsx3...
pip install pyttsx3 --no-cache-dir

echo.
echo Installing tf-keras...
pip install tf-keras --no-cache-dir

echo.
echo ===================================
echo Installation Complete!
echo ===================================
pause