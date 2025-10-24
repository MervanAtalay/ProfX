import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
from collections import deque
import os
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
warnings.filterwarnings('ignore')

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detector with webcam and face detection."""
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.debug_mode = True
        
        # User-configurable delay settings
        self.delay_seconds = 1.5  # Default delay
        self.min_delay = 0.1
        self.max_delay = 5.0
        
        # Single face tracking system with configurable delay
        self.current_emotion = "analyzing..."
        self.current_confidence = 0.0
        self.last_emotion_analysis = 0
        self.emotion_analysis_interval = 0.5  # Analyze emotion every 0.5 second
        self.last_face_position = None
        self.face_lost_frames = 0
        self.max_face_lost_frames = 10  # Reset tracking after 10 frames without face
        
        # Dynamic buffers based on delay setting
        self.emotion_history = deque()
        self.display_emotion_history = deque()
        self.display_emotion = "analyzing..."
        self.display_confidence = 0.0
        
        # Dynamic video frame buffers
        self.frame_buffer = deque()
        self.face_buffer = deque()
        
        # GUI control variables
        self.gui_root = None
        self.delay_var = None
        self.running = False
        self.start_btn = None
        self.stop_btn = None
        
        # Exam-focused emotion mapping
        self.exam_emotions = {
            'happy': 'confident',
            'neutral': 'focused',
            'sad': 'confused',
            'angry': 'frustrated',
            'fear': 'anxiety',
            'surprise': 'curious',
            'disgust': 'bored'
        }
        
        # Initialize dynamic buffers
        self.update_buffer_sizes()
        
        # Test DeepFace availability
        try:
            print("Testing DeepFace...")
            test_img = np.ones((48, 48, 3), dtype=np.uint8) * 128
            DeepFace.analyze(test_img, actions=['emotion'], enforce_detection=False, silent=True)
            print("DeepFace is working correctly!")
        except Exception as e:
            print(f"DeepFace test failed: {e}")
            print("Installing required models...")
    
    def update_buffer_sizes(self):
        """Update buffer sizes based on current delay setting."""
        fps = 30  # Assumed FPS
        
        # Calculate buffer sizes
        frame_buffer_size = max(3, int(self.delay_seconds * fps))  # Minimum 3 frames
        emotion_buffer_size = max(2, int(self.delay_seconds * 2))  # 2 samples per second minimum
        
        # Update maxlen for existing deques by creating new ones
        current_frames = list(self.frame_buffer) if hasattr(self, 'frame_buffer') else []
        current_faces = list(self.face_buffer) if hasattr(self, 'face_buffer') else []
        current_emotions = list(self.emotion_history) if hasattr(self, 'emotion_history') else []
        current_display_emotions = list(self.display_emotion_history) if hasattr(self, 'display_emotion_history') else []
        
        # Create new deques with updated sizes
        self.frame_buffer = deque(current_frames, maxlen=frame_buffer_size)
        self.face_buffer = deque(current_faces, maxlen=frame_buffer_size)
        self.emotion_history = deque(current_emotions, maxlen=emotion_buffer_size)
        self.display_emotion_history = deque(current_display_emotions, maxlen=frame_buffer_size)
        
        if self.debug_mode:
            print(f"Buffer sizes updated - Delay: {self.delay_seconds}s, Frame buffer: {frame_buffer_size}, Emotion buffer: {emotion_buffer_size}")
    
    def create_gui(self):
        """Create the GUI for delay configuration."""
        self.gui_root = tk.Tk()
        self.gui_root.title("Emotion Detection Settings")
        self.gui_root.geometry("500x600")
        self.gui_root.resizable(False, False)
        
        # Set window to be always on top initially
        self.gui_root.attributes('-topmost', True)
        self.gui_root.after_idle(lambda: self.gui_root.attributes('-topmost', False))
        
        # Main frame
        main_frame = ttk.Frame(self.gui_root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Detection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Delay setting
        delay_label = ttk.Label(main_frame, text="Detection Delay (seconds):", 
                               font=("Arial", 12))
        delay_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Delay variable
        self.delay_var = tk.DoubleVar(value=self.delay_seconds)
        
        # Delay scale
        delay_scale = ttk.Scale(main_frame, from_=self.min_delay, to=self.max_delay, 
                               variable=self.delay_var, orient=tk.HORIZONTAL, length=300,
                               command=self.on_delay_change)
        delay_scale.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        # Delay value label
        self.delay_value_label = ttk.Label(main_frame, text=f"{self.delay_seconds:.1f} seconds", 
                                          font=("Arial", 12, "bold"), foreground="blue")
        self.delay_value_label.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Quick preset buttons
        preset_frame = ttk.LabelFrame(main_frame, text="Quick Presets", padding="15")
        preset_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        preset_buttons = [
            ("Real-time (0.1s)", 0.1),
            ("Fast (0.5s)", 0.5),
            ("Medium (1.5s)", 1.5),
            ("Stable (3.0s)", 3.0),
            ("Very Stable (5.0s)", 5.0)
        ]
        
        for i, (text, delay) in enumerate(preset_buttons):
            btn = ttk.Button(preset_frame, text=text, width=18,
                           command=lambda d=delay: self.set_delay_preset(d))
            btn.grid(row=i//2, column=i%2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        preset_frame.columnconfigure(0, weight=1)
        preset_frame.columnconfigure(1, weight=1)
        
        # Control buttons - BÜYÜK VE GÖRÜNÜR
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))
        
        # Create custom style for buttons
        style = ttk.Style()
        style.configure("Large.TButton", font=("Arial", 12, "bold"), padding=(20, 10))
        style.configure("Stop.TButton", font=("Arial", 12, "bold"), padding=(20, 10))
        
        self.start_btn = ttk.Button(button_frame, text="🎥 START DETECTION", 
                                   command=self.start_detection, style="Large.TButton", width=20)
        self.start_btn.grid(row=0, column=0, padx=(0, 10), pady=10)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ STOP DETECTION", 
                                  command=self.stop_detection, style="Stop.TButton", width=20)
        self.stop_btn.grid(row=0, column=1, padx=(10, 0), pady=10)
        self.stop_btn.config(state='disabled')  # Initially disabled
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to start detection", 
                                     font=("Arial", 10), foreground="green")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=(10, 0))
        
        # Info label
        info_text = ("•This is a demo application\n"
                    "• Developed by Arda Sengec \n"
                    "• Licensed under TR-ARGE License\n"
                    "• It is forbidden to use this application without permission\n"
                    "• Click START DETECTION button to begin!")
        
        info_label = ttk.Label(main_frame, text=info_text, font=("Arial", 9), 
                              foreground="gray", justify=tk.LEFT)
        info_label.grid(row=7, column=0, columnspan=1, pady=(20, 0))
        
        # Configure main frame grid weights
        main_frame.columnconfigure(0, weight=1)
        self.gui_root.columnconfigure(0, weight=1)
        self.gui_root.rowconfigure(0, weight=1)
        
        # Handle window close
        self.gui_root.protocol("WM_DELETE_WINDOW", self.on_gui_close)
        
        # Focus on start button
        self.start_btn.focus_set()
    
    def on_delay_change(self, value):
        """Handle delay scale change."""
        new_delay = float(value)
        self.delay_seconds = round(new_delay, 1)
        self.delay_value_label.config(text=f"{self.delay_seconds:.1f} seconds")
        
        # Update buffers if detection is running
        if self.running:
            self.update_buffer_sizes()
            print(f"Delay updated to {self.delay_seconds:.1f} seconds during runtime")
    
    def set_delay_preset(self, delay):
        """Set delay to a preset value."""
        self.delay_var.set(delay)
        self.on_delay_change(delay)
    
    def start_detection(self):
        """Start emotion detection in a separate thread."""
        if not self.running:
            self.running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(text="Detection is starting...", foreground="orange")
            
            detection_thread = threading.Thread(target=self.run_detection, daemon=True)
            detection_thread.start()
            print(f"Starting emotion detection with {self.delay_seconds:.1f}s delay...")
            
            # Update status after a short delay
            self.gui_root.after(2000, lambda: self.status_label.config(
                text="Detection is running! Close this window or click STOP to end.", 
                foreground="green"
            ))
    
    def stop_detection(self):
        """Stop emotion detection."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Detection stopped. Ready to start again.", foreground="red")
        print("Emotion detection stopped.")
    
    def on_gui_close(self):
        """Handle GUI window close."""
        if self.running:
            self.stop_detection()
        self.gui_root.quit()
        self.gui_root.destroy()

    # ... (tüm diğer metodlar aynı kalıyor, sadece GUI kısmını değiştirdim)
    
    def initialize_camera(self, camera_index=0):
        """Initialize the webcam capture."""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_largest_face(self, frame):
        """Detect the largest face in the frame (assuming single person)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques for better detection
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # More strict parameters to reduce false positives
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor for better detection
            minNeighbors=6,    # Higher min neighbors
            minSize=(100, 100),  # Even larger minimum size
            maxSize=(350, 350),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None
        
        # Filter faces by aspect ratio (faces should be roughly square)
        valid_faces = []
        for face in faces:
            x, y, w, h = face
            aspect_ratio = w / h
            if 0.7 <= aspect_ratio <= 1.3:  # Valid face aspect ratio
                valid_faces.append(face)
        
        if not valid_faces:
            return None
        
        # Return the largest valid face
        largest_face = max(valid_faces, key=lambda face: face[2] * face[3])
        
        # If we have a previous face position, prefer faces close to it
        if self.last_face_position is not None:
            last_x, last_y, last_w, last_h = self.last_face_position
            last_center = (last_x + last_w//2, last_y + last_h//2)
            
            closest_face = None
            min_distance = float('inf')
            
            for face in valid_faces:
                x, y, w, h = face
                center = (x + w//2, y + h//2)
                distance = ((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)**0.5
                
                # Also check size similarity
                size_diff = abs((w * h) - (last_w * last_h)) / (last_w * last_h)
                
                if distance < min_distance and distance < 120 and size_diff < 0.5:  # More strict criteria
                    min_distance = distance
                    closest_face = face
            
            if closest_face is not None:
                largest_face = closest_face
        
        if self.debug_mode and self.frame_count % 30 == 0:  # Print less frequently
            print(f"Detected face at: {largest_face}")
        
        return largest_face
    
    def map_to_exam_emotion(self, emotion):
        """Map basic emotions to exam-specific emotions."""
        return self.exam_emotions.get(emotion, emotion)
    
    def analyze_emotion(self, face_img):
        """Analyze emotion using DeepFace."""
        try:
            # Ensure minimum size and good quality
            if face_img.shape[0] < 64 or face_img.shape[1] < 64:
                face_img = cv2.resize(face_img, (128, 128))
            
            # Improve image quality
            face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)  # Enhance contrast
            
            if self.debug_mode and self.frame_count % 30 == 0:
                print(f"Analyzing emotion with face shape: {face_img.shape}")
            
            # Use multiple backends as fallback
            backends = ['opencv', 'retinaface', 'ssd']
            
            for backend in backends:
                try:
                    result = DeepFace.analyze(
                        face_img, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True,
                        detector_backend=backend
                    )
                    
                    # Extract emotion data
                    if isinstance(result, list):
                        emotions = result[0]['emotion']
                    else:
                        emotions = result['emotion']
                    
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    
                    # Map to exam-specific emotion
                    exam_emotion = self.map_to_exam_emotion(dominant_emotion)
                    
                    # Add to history for stability
                    self.emotion_history.append({
                        'emotion': exam_emotion,
                        'original_emotion': dominant_emotion,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
                    
                    # Calculate stable emotion based on current delay
                    self.update_stable_emotion()
                    
                    if self.debug_mode:
                        print(f"Raw emotion: {dominant_emotion} -> Exam emotion: {exam_emotion} ({confidence:.1f}%)")
                        print(f"Stable emotion: {self.current_emotion} ({self.current_confidence:.1f}%)")
                    
                    return True
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"Backend {backend} failed: {e}")
                    continue
            
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error analyzing emotion: {e}")
            return False
    
    def update_stable_emotion(self):
        """Update the stable emotion based on current delay setting."""
        if not self.emotion_history:
            return
        
        # Count recent emotions based on current delay
        current_time = time.time()
        recent_emotions = [
            entry for entry in self.emotion_history 
            if current_time - entry['timestamp'] <= self.delay_seconds
        ]
        
        if not recent_emotions:
            return
        
        # Weight emotions by confidence and recency
        emotion_scores = {}
        total_weight = 0
        
        for entry in recent_emotions:
            emotion = entry['emotion']
            confidence = entry['confidence']
            age = current_time - entry['timestamp']
            
            # Weight by confidence and recency (newer emotions have higher weight)
            recency_weight = max(0.2, 1.0 - (age / self.delay_seconds))
            confidence_weight = confidence / 100.0
            weight = recency_weight * confidence_weight
            
            if emotion not in emotion_scores:
                emotion_scores[emotion] = {'total_weight': 0, 'count': 0, 'total_confidence': 0}
            
            emotion_scores[emotion]['total_weight'] += weight
            emotion_scores[emotion]['count'] += 1
            emotion_scores[emotion]['total_confidence'] += confidence
            total_weight += weight
        
        # Get most weighted emotion
        if emotion_scores and total_weight > 0:
            # Calculate weighted scores
            for emotion in emotion_scores:
                emotion_scores[emotion]['normalized_score'] = emotion_scores[emotion]['total_weight'] / total_weight
                emotion_scores[emotion]['avg_confidence'] = emotion_scores[emotion]['total_confidence'] / emotion_scores[emotion]['count']
            
            # Select emotion with highest weighted score
            best_emotion = max(emotion_scores, key=lambda x: emotion_scores[x]['normalized_score'])
            avg_confidence = emotion_scores[best_emotion]['avg_confidence']
            
            # Update current emotion (threshold based on delay)
            min_count = 1 if self.delay_seconds < 1.0 else 2
            if emotion_scores[best_emotion]['count'] >= min_count:
                self.current_emotion = best_emotion
                self.current_confidence = avg_confidence
                
                # Add to display history with current delay
                self.display_emotion_history.append({
                    'emotion': best_emotion,
                    'confidence': avg_confidence,
                    'timestamp': time.time()
                })
    
    def update_display_emotion(self):
        """Update the emotion to display with current delay."""
        if not self.display_emotion_history:
            return
        
        current_time = time.time()
        
        # Find emotion from delay_seconds ago
        target_time = current_time - self.delay_seconds
        closest_entry = None
        min_time_diff = float('inf')
        
        for entry in self.display_emotion_history:
            time_diff = abs(entry['timestamp'] - target_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_entry = entry
        
        # Tolerance based on delay (more tolerance for shorter delays)
        tolerance = max(0.3, self.delay_seconds * 0.5)
        if closest_entry and min_time_diff < tolerance:
            self.display_emotion = closest_entry['emotion']
            self.display_confidence = closest_entry['confidence']
    
    def add_frame_to_buffer(self, frame, face_data):
        """Add current frame and face data to buffer with timestamp."""
        frame_data = {
            'frame': frame.copy(),
            'face': face_data,
            'timestamp': time.time()
        }
        self.frame_buffer.append(frame_data)
        self.face_buffer.append(face_data)
    
    def get_delayed_frame(self):
        """Get frame from delay_seconds ago."""
        if not self.frame_buffer:
            return None, None
        
        current_time = time.time()
        target_time = current_time - self.delay_seconds
        
        # Find the frame closest to delay_seconds ago
        closest_frame_data = None
        min_time_diff = float('inf')
        
        for frame_data in self.frame_buffer:
            time_diff = abs(frame_data['timestamp'] - target_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_frame_data = frame_data
        
        # Tolerance based on delay
        tolerance = max(0.2, self.delay_seconds * 0.3)
        if closest_frame_data and min_time_diff < tolerance:
            return closest_frame_data['frame'], closest_frame_data['face']
        
        return None, None
    
    def draw_face_info(self, frame, face_data):
        """Draw rectangle around face and emotion text."""
        if face_data is None:
            return
        
        x, y, w, h = face_data
        
        # Update display emotion first
        self.update_display_emotion()
        
        # Draw rectangle around face with color based on exam emotion
        emotion_colors = {
            'confident': (0, 255, 0),      # Green
            'confused': (0, 165, 255),     # Orange
            'anxiety': (0, 0, 255),        # Red
            'frustrated': (0, 0, 139),     # Dark Red
            'focused': (128, 128, 128),    # Gray
            'curious': (255, 255, 0),      # Cyan
            'bored': (128, 0, 128),        # Purple
            'analyzing': (255, 255, 255)   # White
        }
        
        color = emotion_colors.get(self.display_emotion, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Prepare text
        emotion_text = f"{self.display_emotion.capitalize()}"
        confidence_text = f"({self.display_confidence:.1f}%)" if self.display_confidence > 0 else ""
        
        # Calculate text position
        text_y = y - 15 if y - 15 > 15 else y + h + 25
        
        # Draw emotion text background
        text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(frame, (x, text_y - 35), (x + text_size[0] + 15, text_y + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, text_y - 35), (x + text_size[0] + 15, text_y + 10), color, 2)
        
        # Draw emotion text
        cv2.putText(frame, emotion_text, (x + 8, text_y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw confidence if available
        if self.display_confidence > 0:
            cv2.putText(frame, confidence_text, (x + 8, text_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw delay indicator
        delay_text = f"{self.delay_seconds:.1f}s delay"
        cv2.putText(frame, delay_text, (x + w - 100, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def calculate_fps(self):
        """Calculate and return current FPS."""
        current_time = time.time()
        self.fps_counter.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.fps_counter) > 1:
            avg_frame_time = sum(self.fps_counter) / len(self.fps_counter)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            return fps
        return 0
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter and system info on the frame."""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw system status with dynamic buffer sizes
        status_text = f"Frames: {len(self.frame_buffer)}/{self.frame_buffer.maxlen} | Emotions: {len(self.emotion_history)}/{self.emotion_history.maxlen}"
        cv2.putText(frame, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw delay info
        delay_info = f"Delay: {self.delay_seconds:.1f}s | Press 'h' for help"
        cv2.putText(frame, delay_info, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw buffer building warning
        min_buffer_size = max(3, int(self.delay_seconds * 15))  # 50% of expected buffer
        if len(self.frame_buffer) < min_buffer_size:
            buffer_text = f"Building buffer... ({len(self.frame_buffer)}/{self.frame_buffer.maxlen})"
            cv2.putText(frame, buffer_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw exam emotion legend
        legend_y = frame.shape[0] - 120
        cv2.putText(frame, "Exam Emotions:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Confident | Confused | Anxiety", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "Frustrated | Focused | Curious | Bored", (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def show_help(self):
        """Show help message in a popup."""
        help_text = """
Emotion Detection Controls:

GUI Controls:
- Use the slider to adjust delay (0.1s - 5.0s)
- Click preset buttons for quick settings
- Changes apply in real-time

Keyboard Controls:
- 'q': Quit detection
- 'c': Switch camera
- 'd': Toggle debug mode
- 'h': Show this help

Delay Settings:
- 0.1s: Near real-time (less stable)
- 0.5s: Fast response (moderately stable)
- 1.5s: Balanced (good stability)
- 3.0s: Stable (very reliable)
- 5.0s: Maximum stability (slow response)
"""
        # Create a simple popup window
        popup = tk.Toplevel(self.gui_root)
        popup.title("Help")
        popup.geometry("400x350")
        popup.resizable(False, False)
        
        text_widget = tk.Text(popup, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        close_btn = ttk.Button(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)
    
    def run_detection(self):
        """Main loop for emotion detection."""
        print(f"Starting emotion detection with {self.delay_seconds:.1f}s delay...")
        print("- Video delay: Configurable")
        print("- Emotion analysis: Every 0.5 second")
        print("- Emotion stability: Based on delay setting")
        print("- Display delay: User configurable")
        print("- Exam emotions: confident, confused, anxiety, frustrated, focused, curious, bored")
        print("Press 'h' in video window for help")
        
        camera_index = 0
        
        if not self.initialize_camera(camera_index):
            self.running = False
            return
        
        try:
            while self.running:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect the largest face in current frame
                face = self.detect_largest_face(frame)
                
                if face is not None:
                    x, y, w, h = face
                    self.last_face_position = face
                    self.face_lost_frames = 0
                    
                    # Extract face region for emotion analysis
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Analyze emotion periodically (on current frame)
                    current_time = time.time()
                    if current_time - self.last_emotion_analysis >= self.emotion_analysis_interval:
                        # Start emotion analysis in a separate thread
                        emotion_thread = threading.Thread(
                            target=self.analyze_emotion, 
                            args=(face_img,)
                        )
                        emotion_thread.daemon = True
                        emotion_thread.start()
                        self.last_emotion_analysis = current_time
                else:
                    self.face_lost_frames += 1
                    
                    # Reset tracking if face is lost for too long
                    if self.face_lost_frames > self.max_face_lost_frames:
                        self.last_face_position = None
                        if self.face_lost_frames == self.max_face_lost_frames + 1:  # Only print once
                            print("Face tracking reset due to prolonged face loss")
                
                # Add current frame and face data to buffer
                self.add_frame_to_buffer(frame, face)
                
                # Get delayed frame based on current delay setting
                display_frame, delayed_face = self.get_delayed_frame()
                
                if display_frame is not None:
                    # Use the delayed frame for display
                    display_frame = display_frame.copy()
                    
                    # Draw face information on delayed frame
                    if delayed_face is not None:
                        self.draw_face_info(display_frame, delayed_face)
                    
                    # Calculate and display FPS
                    fps = self.calculate_fps()
                    self.draw_fps(display_frame, fps)
                    
                    # Display the delayed frame
                    cv2.imshow(f'Exam Emotion Detection ({self.delay_seconds:.1f}s delay)', display_frame)
                else:
                    # Show current frame with buffer building message if no delayed frame available
                    current_display = frame.copy()
                    fps = self.calculate_fps()
                    self.draw_fps(current_display, fps)
                    cv2.imshow(f'Exam Emotion Detection ({self.delay_seconds:.1f}s delay)', current_display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c'):
                    # Switch camera
                    camera_index = (camera_index + 1) % 3
                    print(f"Switching to camera {camera_index}...")
                    self.cap.release()
                    # Clear buffers when switching camera
                    self.frame_buffer.clear()
                    self.face_buffer.clear()
                    self.emotion_history.clear()
                    self.display_emotion_history.clear()
                    if not self.initialize_camera(camera_index):
                        camera_index = 0
                        self.initialize_camera(camera_index)
                elif key == ord('d'):
                    # Toggle debug mode
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('h'):
                    # Show help
                    self.show_help()
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Clean up resources
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped and resources cleaned up.")

def main():
    """Main function to run the emotion detector with GUI."""
    print("=" * 60)
    print("Configurable Exam Emotion Detection System")
    print("=" * 60)
    
    # Create detector and GUI
    detector = EmotionDetector()
    detector.create_gui()
    
    # Start GUI main loop
    try:
        detector.gui_root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted")
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    main()