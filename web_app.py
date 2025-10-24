from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from deepface import DeepFace
import base64
import json

app = FastAPI(title="AI Teacher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <head><title>AI Teacher</title></head>
        <body>
            <h1>🤖 AI Virtual Teacher</h1>
            <p>API is running!</p>
            <video id="video" width="640" height="480" autoplay></video>
            <p id="emotion">Emotion: detecting...</p>
            
            <script>
                const video = document.getElementById('video');
                const emotionText = document.getElementById('emotion');
                
                // Start camera
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => video.srcObject = stream);
                
                // WebSocket connection
                const ws = new WebSocket(`ws://${window.location.host}/ws/emotion`);
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    emotionText.textContent = 'Emotion: ' + data.emotion;
                };
                
                // Send frames
                setInterval(() => {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    
                    const imageData = canvas.toDataURL('image/jpeg', 0.7);
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(imageData);
                    }
                }, 1000);
            </script>
        </body>
    </html>
    """)

@app.websocket("/ws/emotion")
async def websocket_emotion(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Decode base64 image
            img_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detect emotion
            try:
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                emotions = result[0]['emotion']
                dominant = max(emotions, key=emotions.get)
                
                await websocket.send_json({
                    'emotion': dominant,
                    'all_emotions': emotions
                })
            except Exception as e:
                await websocket.send_json({
                    'emotion': 'error',
                    'message': str(e)
                })
                
    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)