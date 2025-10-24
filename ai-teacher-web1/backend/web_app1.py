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
import base64  # ✅ ALWAYS IMPORT (needed for WebSocket in both dev and prod)

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
                    {"question": "If you have 4 apples and get 2 more, how many do you have?", "answer": "6", "alternatives": ["six"]},
                    {"question": "What is 10 plus 5?", "answer": "15", "alternatives": ["fifteen"]}
                ]
            },
            {
                "title": "Subtraction",
                "explanation": "Subtraction is taking away one number from another. For example, 5 minus 2 equals 3.",
                "questions": [
                    {"question": "What is 10 minus 4?", "answer": "6", "alternatives": ["six"]},
                    {"question": "If you have 8 cookies and eat 3, how many are left?", "answer": "5", "alternatives": ["five"]}
                ]
            }
        ]
    },
    "english": {
        "title": "Basic English",
        "topics": [
            {
                "title": "Colors",
                "explanation": "Colors are what we see around us. The sky is blue, grass is green, and the sun is yellow.",
                "questions": [
                    {"question": "What color is the sky?", "answer": "blue", "alternatives": []},
                    {"question": "What color is grass?", "answer": "green", "alternatives": []}
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
                    {"question": "4 elmanız varsa ve 2 tane daha alırsanız, kaç elmanız olur?", "answer": "6", "alternatives": ["altı"]},
                    {"question": "10 artı 5 kaç eder?", "answer": "15", "alternatives": ["on beş"]}
                ]
            },
            {
                "title": "Çıkarma",
                "explanation": "Çıkarma, bir sayıdan başka bir sayıyı çıkarmaktır. Örneğin, 5 eksi 2 eşittir 3.",
                "questions": [
                    {"question": "10 eksi 4 kaç eder?", "answer": "6", "alternatives": ["altı"]},
                    {"question": "8 kurabiyeniz varsa ve 3 tanesini yerseniz, kaç tane kalır?", "answer": "5", "alternatives": ["beş"]}
                ]
            }
        ]
    },
    "ingilizce": {
        "title": "Temel İngilizce",
        "topics": [
            {
                "title": "Renkler",
                "explanation": "Renkler etrafımızda gördüğümüz şeylerdir. Gökyüzü mavidir, çimen yeşildir ve güneş sarıdır.",
                "questions": [
                    {"question": "Gökyüzü hangi renktir?", "answer": "mavi", "alternatives": ["blue"]},
                    {"question": "Çimen hangi renktir?", "answer": "yeşil", "alternatives": ["green"]}
                ]
            }
        ]
    }
}

@app.get("/")
async def root():
    """Serve the main HTML page"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "frontend", "index.html")
    
    try:
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Error</title></head>
                <body>
                    <h1>❌ Frontend Not Found</h1>
                    <p>Looking for: {html_path}</p>
                    <p>Create the frontend/index.html file</p>
                </body>
            </html>
            """,
            status_code=500
        )

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "deepface": DEEPFACE_AVAILABLE,
        "opencv": OPENCV_AVAILABLE,
        "gemini": gemini_client is not None,
        "gemini_model": gemini_model_name,
        "production_mode": IS_PRODUCTION,
        "active_sessions": len(active_sessions)
    }

@app.get("/api/lessons")
async def get_lessons(language: str = "en"):
    """Get available lessons"""
    lessons = LESSONS_TR if language == "tr" else LESSONS_EN
    return {
        "lessons": {
            key: {
                "title": value["title"],
                "topic_count": len(value["topics"])
            }
            for key, value in lessons.items()
        }
    }

@app.get("/api/lesson/{lesson_name}")
async def get_lesson(lesson_name: str, language: str = "en"):
    """Get specific lesson details"""
    lessons = LESSONS_TR if language == "tr" else LESSONS_EN
    
    if lesson_name not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    return lessons[lesson_name]

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

@app.post("/api/lesson/start")
async def start_lesson(data: dict):
    """Start a new lesson"""
    session_id = data.get('session_id')
    lesson_name = data.get('lesson_name')
    language = data.get('language', 'en')
    
    lessons = LESSONS_TR if language == "tr" else LESSONS_EN
    
    if lesson_name not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Initialize session progress
    if session_id not in active_sessions:
        active_sessions[session_id] = {}
    
    active_sessions[session_id]['lesson'] = {
        'name': lesson_name,
        'language': language,
        'topic_index': 0,
        'question_index': 0,
        'data': lessons[lesson_name]
    }
    active_sessions[session_id]['last_emotion'] = 'focused'
    
    # Get first topic
    first_topic = lessons[lesson_name]['topics'][0]
    
    return {
        "status": "started",
        "lesson_name": lesson_name,
        "lesson_title": lessons[lesson_name]['title'],
        "topic": first_topic['title'],
        "explanation": first_topic['explanation']
    }

@app.post("/api/lesson/next_question")
async def get_next_question(data: dict):
    """Get next question in current lesson"""
    session_id = data.get('session_id')
    
    if session_id not in active_sessions or 'lesson' not in active_sessions[session_id]:
        raise HTTPException(status_code=400, detail="No active lesson")
    
    session = active_sessions[session_id]
    lesson_data = session['lesson']['data']
    topic_index = session['lesson']['topic_index']
    question_index = session['lesson']['question_index']
    
    topics = lesson_data['topics']
    
    if topic_index >= len(topics):
        return {
            "completed": True,
            "message": "Congratulations! You completed the lesson!" if session['lesson']['language'] == 'en' 
                      else "Tebrikler! Dersi tamamladınız!"
        }
    
    current_topic = topics[topic_index]
    questions = current_topic['questions']
    
    if question_index >= len(questions):
        # Move to next topic
        session['lesson']['topic_index'] += 1
        session['lesson']['question_index'] = 0
        
        if session['lesson']['topic_index'] >= len(topics):
            return {
                "completed": True,
                "message": "Congratulations! You completed the lesson!" if session['lesson']['language'] == 'en'
                          else "Tebrikler! Dersi tamamladınız!"
            }
        
        # Get next topic
        next_topic = topics[session['lesson']['topic_index']]
        return {
            "new_topic": True,
            "topic": next_topic['title'],
            "explanation": next_topic['explanation']
        }
    
    question = questions[question_index]
    session['lesson']['question_index'] += 1
    
    return {
        "question": question['question'],
        "question_data": question,
        "topic": current_topic['title'],
        "progress": {
            "topic": topic_index + 1,
            "total_topics": len(topics),
            "question": question_index + 1,
            "total_questions": len(questions)
        }
    }

@app.websocket("/ws/emotion/{session_id}")
async def websocket_emotion(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time emotion detection"""
    await websocket.accept()
    print(f"✅ WebSocket connected: {session_id}")
    
    if session_id not in active_sessions:
        active_sessions[session_id] = {}
    
    active_sessions[session_id]['websocket'] = websocket
    active_sessions[session_id]['last_emotion'] = 'focused'
    active_sessions[session_id]['connected_at'] = datetime.now()
    
    try:
        while True:
            # Receive data from frontend
            data = await websocket.receive_text()
            
            # PRODUCTION MODE - Return default emotion (no ML processing)
            if IS_PRODUCTION or not DEEPFACE_AVAILABLE or not OPENCV_AVAILABLE:
                await websocket.send_json({
                    'emotion': 'focused',
                    'confidence': 0.75,
                    'all_emotions': {'focused': 75.0, 'neutral': 25.0},
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Production mode - emotion detection disabled'
                })
                await asyncio.sleep(2)
                continue
            
            # DEVELOPMENT MODE - Try emotion detection with DeepFace
            try:
                import cv2
                import numpy as np
                from deepface import DeepFace
                
                # Decode base64 image
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
                    'emotion': 'focused',
                    'confidence': 0.7,
                    'all_emotions': {'focused': 70.0, 'neutral': 30.0}
                })
            
            await asyncio.sleep(0.1)
            
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
            emotion = active_sessions[session_id].get('last_emotion', 'neutral')
        
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
            except Exception as e:
                print(f"Gemini error: {e}")
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
            "needs_re_explanation": not is_correct or (is_correct and emotion_confidence < 0.5)
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
            emotion = active_sessions[session_id].get('last_emotion', 'neutral')
        
        if gemini_client and gemini_model_name:
            lang_instr = "Respond in Turkish" if language == "tr" else "Respond in English"
            prompt = f"""You are a friendly AI teacher assistant for elementary students.

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
            except Exception as e:
                print(f"Gemini error: {e}")
                ai_response = "I'm here to help you learn! What would you like to know?" if language == "en" else "Sana öğrenmende yardımcı olmak için buradayım! Ne öğrenmek istersin?"
        else:
            ai_response = "That's interesting! Tell me more." if language == "en" else "İlginç! Bana daha fazla anlat."
        
        return {"response": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    """Get student progress"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                   AVG(confidence) as avg_confidence
            FROM sessions 
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "total_questions": result[0] or 0,
            "correct_answers": result[1] or 0,
            "accuracy": (result[1] / result[0] * 100) if result[0] > 0 else 0,
            "avg_confidence": result[2] or 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🚀 AI Virtual Teacher - Web API")
    print(f"Server: http://localhost:8000")
    print(f"DeepFace (Emotion): {'✅ Enabled' if DEEPFACE_AVAILABLE else '❌ Disabled'}")
    print(f"Gemini AI: {'✅ Enabled' if gemini_client else '❌ Disabled'}")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
