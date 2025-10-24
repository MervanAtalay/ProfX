# Real-time Emotion Detection
#To activate : .\ai_teacher_env\Scripts\activate
# ssh root@159.198.77.153 (melis123)
# project location on remote : /var/www/ai_teacher
# https://github.com/MervanAtalay/ProfX.git
A Python application that uses your computer's webcam to detect human faces and analyze their emotions in real-time using OpenCV and DeepFace.

## Features

- ✅ Real-time face detection using webcam
- 🎭 Emotion analysis (happy, sad, angry, surprised, neutral, fear, disgust)
- 📊 Live emotion display on detected faces
- 🎯 Face bounding boxes
- 📈 FPS counter
- 📷 Camera switching support
- 🛡️ Graceful error handling

## Requirements

- Python 3.9 or newer
- Webcam/Camera
- Dependencies listed in `requirements.txt`

## Quick Setup

### Option 1: Automatic Setup
```bash
python setup_project.py
```

### Option 2: Manual Setup
1. Create virtual environment:
```bash
python -m venv emotion_detection_env
```

2. Activate virtual environment:
```bash
# Windows
emotion_detection_env\Scripts\activate

# Linux/Mac
source emotion_detection_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your virtual environment is activated
2. Run the emotion detector:
```bash
python emotion_detector.py
```

## Controls

- **Q**: Quit the application
- **C**: Switch between available cameras

## Technical Details

### Libraries Used
- **OpenCV**: For webcam access and face detection
- **DeepFace**: For emotion analysis
- **TensorFlow**: Backend for DeepFace
- **NumPy**: For numerical operations

### Performance Features
- Emotion analysis runs every 10th frame for better performance
- Multi-threading for non-blocking emotion analysis
- Emotion caching to reduce computation
- Automatic cleanup of old cache entries

### Supported Emotions
- Happy
- Sad
- Angry
- Surprised
- Neutral
- Fear
- Disgust

## Troubleshooting

### Camera Issues
- Make sure no other application is using the camera
- Try pressing 'c' to switch to a different camera
- Check camera permissions in your system settings

### Performance Issues
- Close other applications that might be using system resources
- Reduce the frame analysis frequency by changing the modulo value in the code
- Ensure good lighting for better face detection

### Installation Issues
- Make sure you have Python 3.9+
- Try installing dependencies one by one if batch installation fails
- On some systems, you might need to install additional system packages for OpenCV

## Project Structure

```
emotion_detector.py     # Main application file
requirements.txt        # Python dependencies
setup_project.py       # Automated setup script
README.md              # This file
```

## License

This project is for educational and personal use.

self.exam_emotions = {
    'happy': 'confident',     # Mutlu -> Kendinden emin
    'neutral': 'focused',     # Nötr -> Odaklanmış
    'sad': 'confused',        # Üzgün -> Kafası karışık
    'angry': 'frustrated',    # Sinirli -> Hayal kırıklığı
    'fear': 'anxiety',        # Korku -> Kaygılı
    'surprise': 'curious',    # Şaşırmış -> Meraklı
    'disgust': 'bored'        # İğrenme -> Sıkılmış
}

3. Sınav Emotion Renkleri

Confident: Yeşil (başarılı hissetme)
Confused: Turuncu (kafası karışık)
Anxiety: Kırmızı (kaygı/stres)
Frustrated: Koyu kırmızı (hayal kırıklığı)
Focused: Gri (odaklanmış)
Curious: Cyan (meraklı)
Bored: Mor (sıkılmış)

4. Performans İyileştirmeleri
Daha düşük threshold (count >= 1)
Faster analysis interval (0.5 saniye)
Daha dar tolerance (0.3 saniye)

5. UI İyileştirmeleri
Exam emotion legend eklendi
"1.5s delay" indicator
Buffer status güncellemesi