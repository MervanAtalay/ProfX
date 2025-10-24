# Production mode imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import sqlite3
import asyncio
import platform

# Check if production (Linux) or development (Windows)
IS_PRODUCTION = platform.system() == "Linux"

# Disable heavy ML libraries on production
DEEPFACE_AVAILABLE = False
OPENCV_AVAILABLE = False

if IS_PRODUCTION:
    print("🚀 PRODUCTION MODE - Running on VPS")
    print("⚠️ Emotion detection DISABLED (saves resources)")
else:
    print("💻 DEVELOPMENT MODE")
    try:
        import cv2
        import numpy as np
        import base64
        OPENCV_AVAILABLE = True
        print("✅ OpenCV available")
    except:
        print("⚠️ OpenCV not available")
    
    try:
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
        print("✅ DeepFace available")
    except:
        print("⚠️ DeepFace not available")

# Import Gemini (works on both)
try:
    from google import genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini library available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini library not available")

app = FastAPI(title="AI Virtual Teacher API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_sessions = {}

# Initialize Gemini if available
gemini_client = None
gemini_model_name = None

if GEMINI_AVAILABLE:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        try:
            with open('../gemini_config.json', 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key')
        except:
            pass
    
    if api_key:
        try:
            gemini_client = genai.Client(api_key=api_key)
            for model_name in ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']:
                try:
                    gemini_client.models.generate_content(model=model_name, contents="Hi")
                    gemini_model_name = model_name
                    print(f"✅ Gemini initialized: {model_name}")
                    break
                except:
                    continue
        except Exception as e:
            print(f"⚠️ Gemini init error: {e}")

# Database setup
DB_PATH = "student_progress.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            lesson_name TEXT,
            topic_name TEXT,
            question TEXT,
            answer TEXT,
            correct BOOLEAN,
            confidence REAL,
            face_emotion TEXT,
            language TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            grade TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_database()

# Lessons database
LESSONS_EN = {
    "mathematics": {
        "title": "Basic Mathematics",
        "topics": [
            {
                "title": "Addition",
                "explanation": "Addition is combining two or more numbers to get their sum. For example, 2 plus 3 equals 5.",
                "questions": [
                    {"question": "What is 5 plus 3?", "answer": "8", "alternatives": ["eight"]},
                    {"question": "If you have 4 apples and get 2 more, how many do you have?", "answer": "6", "alternatives": ["six"]}
                ]
            }
        ]
    }
}

LESSONS_TR = {
    "matematik": {
        "title": "Temel Matematik",
        "topics": [
            {
                "title": "Toplama",
                "explanation": "Toplama, iki veya daha fazla sayıyı birleştirerek toplamlarını bulmaktır. Örneğin, 2 artı 3 eşittir 5.",
                "questions": [
                    {"question": "5 artı 3 kaç eder?", "answer": "8", "alternatives": ["sekiz"]},
                    {"question": "4 elmanız varsa ve 2 tane daha alırsanız, kaç elmanız olur?", "answer": "6", "alternatives": ["altı"]}
                ]
            }
        ]
    }
}

# Replace with:
@app.get("/")
async def root():
    """Serve the main HTML page"""
    import os
    
    # Get the absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "frontend", "index.html")
    
    try:
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Error</title></head>
                <body>
                    <h1>❌ Frontend Not Found</h1>
                    <p>Looking for: {}</p>
                    <p>Create the frontend/index.html file</p>
                </body>
            </html>
            """.format(html_path),
            status_code=500
        )
@app.post("/api/profile")
async def save_profile(data: dict):
    """Save student profile"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO profiles (name, age, grade)
            VALUES (?, ?, ?)
        ''', (data.get('name'), data.get('age'), data.get('grade')))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/emotion/{session_id}")
async def websocket_emotion(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time emotion detection"""
    await websocket.accept()
    print(f"✅ WebSocket connected: {session_id}")
    
    active_sessions[session_id] = {
        'websocket': websocket,
        'last_emotion': 'neutral',
        'connected_at': datetime.now()
    }
    
    try:
        while True:
            # Receive base64 encoded image from frontend
            data = await websocket.receive_text()
            
            try:
                # Decode image
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({'emotion': 'no_frame', 'error': 'Invalid frame'})
                    continue
                
                # Detect emotion using DeepFace
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(result, list):
                    emotions = result[0]['emotion']
                else:
                    emotions = result['emotion']
                
                dominant = max(emotions, key=emotions.get)
                
                # Map to teaching-relevant emotions
                emotion_map = {
                    'happy': 'confident',
                    'neutral': 'focused',
                    'sad': 'confused',
                    'angry': 'frustrated',
                    'fear': 'anxiety',
                    'surprise': 'curious',
                    'disgust': 'bored'
                }
                
                teaching_emotion = emotion_map.get(dominant, dominant)
                active_sessions[session_id]['last_emotion'] = teaching_emotion
                
                # Send back emotion data
                await websocket.send_json({
                    'emotion': teaching_emotion,
                    'confidence': float(emotions[dominant]),
                    'all_emotions': {k: float(v) for k, v in emotions.items()},
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Emotion detection error: {e}")
                await websocket.send_json({
                    'emotion': 'analyzing',
                    'error': str(e)
                })
            
    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected: {session_id}")
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/api/evaluate_answer")
async def evaluate_answer(data: dict):
    """Evaluate student's answer"""
    try:
        question_data = data.get('question_data')
        student_answer = data.get('answer', '').strip().lower()
        session_id = data.get('session_id')
        language = data.get('language', 'en')
        
        if not question_data or not student_answer:
            return {"correct": False, "response": "No answer provided"}
        
        # Get current emotion
        emotion = 'neutral'
        if session_id in active_sessions:
            emotion = active_sessions[session_id]['last_emotion']
        
        # Check answer
        correct_answer = question_data['answer'].lower()
        alternatives = [alt.lower() for alt in question_data.get('alternatives', [])]
        
        is_correct = (correct_answer in student_answer or 
                     any(alt in student_answer for alt in alternatives))
        
        # Calculate confidence
        text_confidence = 0.8 if is_correct else 0.3
        
        emotion_confidence = 0.5
        if emotion in ['confident', 'focused', 'happy']:
            emotion_confidence = 0.9
        elif emotion in ['confused', 'anxiety', 'frustrated']:
            emotion_confidence = 0.2
        
        overall_confidence = (text_confidence * 0.6) + (emotion_confidence * 0.4)
        
        # Generate response with Gemini if available
        if gemini_client and gemini_model_name:
            try:
                lang_instr = "Respond in Turkish" if language == "tr" else "Respond in English"
                prompt = f"""You are a patient AI teacher for elementary students.

Question: {question_data['question']}
Student's Answer: {student_answer}
Correct: {is_correct}
Student's Emotion: {emotion}

{lang_instr}. Provide a brief, encouraging response (1-2 sentences):
- If correct: praise warmly
- If incorrect: gently guide without giving full answer
- Keep it simple and supportive

Response:"""
                
                response_obj = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt
                )
                ai_response = response_obj.text.strip()
            except:
                ai_response = "Great job!" if is_correct else "Good try! Let's think about this again."
        else:
            ai_response = "Excellent!" if is_correct else "Not quite, but keep trying!"
        
        # Log to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, lesson_name, topic_name, question, answer, correct, confidence, face_emotion, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            data.get('lesson_name', ''),
            data.get('topic_name', ''),
            question_data['question'],
            student_answer,
            is_correct,
            overall_confidence,
            emotion,
            language
        ))
        conn.commit()
        conn.close()
        
        return {
            "correct": is_correct,
            "confidence": overall_confidence,
            "emotion": emotion,
            "response": ai_response,
            "needs_re_explanation": is_correct and emotion_confidence < 0.5
        }
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(data: dict):
    """Handle general educational chat"""
    try:
        message = data.get('message', '')
        session_id = data.get('session_id')
        language = data.get('language', 'en')
        
        emotion = 'neutral'
        if session_id in active_sessions:
            emotion = active_sessions[session_id]['last_emotion']
        
        if gemini_client and gemini_model_name:
            lang_instr = "Respond in Turkish" if language == "tr" else "Respond in English"
            prompt = f"""You are a friendly AI teacher assistant.

Student's Message: {message}
Student's Emotion: {emotion}

{lang_instr}. Provide a helpful, age-appropriate response (2-3 sentences max).
Be supportive and educational.

Response:"""
            
            try:
                response_obj = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt
                )
                ai_response = response_obj.text.strip()
            except:
                ai_response = "I'm here to help you learn! What would you like to know?" if language == "en" else "Sana öğrenmende yardımcı olmak için buradayım! Ne öğrenmek istersin?"
        else:
            ai_response = "That's interesting! Tell me more." if language == "en" else "İlginç! Bana daha fazla anlat."
        
        return {"response": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🚀 AI Virtual Teacher - Web API")
    print(f"Server: http://localhost:8000")
    print(f"Gemini AI: {'✅ Enabled' if gemini_client else '❌ Disabled'}")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")