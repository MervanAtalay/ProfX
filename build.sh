# filepath: c:\Users\320200432\Desktop\PY-EMR\build.sh
#!/bin/bash

echo "🚀 Building Emotion Detector for macOS..."

# Temizleme
rm -rf build/ dist/ *.spec

# PyInstaller ile build
pyinstaller \
    --name "EmotionDetector" \
    --onedir \
    --windowed \
    --noconfirm \
    --clean \
    --hidden-import deepface \
    --hidden-import tensorflow \
    --hidden-import keras \
    --hidden-import cv2 \
    --hidden-import tkinter \
    --hidden-import PIL \
    --hidden-import numpy \
    --hidden-import pandas \
    --exclude-module matplotlib \
    --exclude-module torch \
    emotion_detector.py

# Kamera izni için Info.plist güncelle
if [ -d "dist/EmotionDetector.app" ]; then
    echo "✅ App bundle created successfully!"
    
    # Info.plist oluştur
    cat > "dist/EmotionDetector.app/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Emotion Detector</string>
    <key>CFBundleExecutable</key>
    <string>EmotionDetector</string>
    <key>CFBundleIdentifier</key>
    <string>com.ardasengec.emotiondetector</string>
    <key>CFBundleName</key>
    <string>EmotionDetector</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>NSCameraUsageDescription</key>
    <string>This application needs camera access to detect emotions.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
</dict>
</plist>
EOF

    echo "✅ Info.plist created with camera permissions"
    echo "🎉 EmotionDetector.app ready!"
    echo "📁 Location: $(pwd)/dist/EmotionDetector.app"
else
    echo "❌ Build failed - .app not found"
fi