import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import time
import os
import json
import pickle
import numpy as np
from PIL import Image, ImageTk
import datetime
import shutil
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Loading PERFECT Face Detection & Recognition System...")

class AdvancedFaceRecognition:
    """
    Advanced face recognition using LBPH (Local Binary Patterns Histograms)
    - Most accurate face recognition algorithm in OpenCV
    - Works well in different lighting conditions
    - Handles multiple faces perfectly
    """
    def __init__(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Face recognition parameters
        self.recognition_threshold = 70  # Lower = more strict
        self.min_face_size = 80
        self.training_data = []
        self.training_labels = []
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        self.load_training_data()
    
    def load_training_data(self):
        """Load trained face recognition model"""
        try:
            if os.path.exists('face_recognition_model.yml'):
                self.face_recognizer.read('face_recognition_model.yml')
                
            if os.path.exists('face_labels.pkl'):
                with open('face_labels.pkl', 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
            if os.path.exists('training_data.pkl'):
                with open('training_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.training_data = data.get('data', [])
                    self.training_labels = data.get('labels', [])
                    
                if len(self.training_data) > 0:
                    self.is_trained = True
                    print(f"✅ Loaded trained model with {len(set(self.training_labels))} persons")
                    
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            self.training_data = []
            self.training_labels = []
    
    def save_training_data(self):
        """Save trained model and data"""
        try:
            self.face_recognizer.write('face_recognition_model.yml')
            
            with open('face_labels.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
                
            with open('training_data.pkl', 'wb') as f:
                pickle.dump({
                    'data': self.training_data,
                    'labels': self.training_labels
                }, f)
                
            return True
        except Exception as e:
            print(f"❌ Error saving training data: {e}")
            return False
    
    def detect_faces_advanced(self, frame):
        """
        Advanced face detection with multiple techniques
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better detection in different lighting
        gray_eq = cv2.equalizeHist(gray)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=8,  # Increased for better accuracy
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter faces by aspect ratio and size
        valid_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            # Valid human face aspect ratio is typically between 0.7 and 1.3
            if 0.7 <= aspect_ratio <= 1.3 and w >= self.min_face_size:
                valid_faces.append((x, y, w, h))
        
        return valid_faces
    
    def preprocess_face(self, face_region):
        """
        Preprocess face for better recognition
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            standardized = cv2.resize(gray, (200, 200))
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(standardized)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            return blurred
        except Exception as e:
            print(f"Face preprocessing error: {e}")
            return None
    
    def register_face(self, frame, name, capture_multiple=True):
        """
        Register face with multiple samples for better accuracy
        """
        try:
            faces = self.detect_faces_advanced(frame)
            
            if len(faces) == 0:
                return False, "No face detected. Please ensure good lighting and clear visibility."
            elif len(faces) > 1:
                return False, "Multiple faces detected. Please ensure only one person is in frame."
            else:
                x, y, w, h = faces[0]
                face_region = frame[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = self.preprocess_face(face_region)
                if processed_face is None:
                    return False, "Could not process face image."
                
                # Add to training data
                self.training_data.append(processed_face)
                self.training_labels.append(name)
                
                # If we have enough samples, train the model
                if len(self.training_data) >= 2:  # At least 2 samples per person
                    self.train_model()
                
                self.save_training_data()
                return True, f"Face '{name}' registered successfully! Samples: {len([l for l in self.training_labels if l == name])}"
                
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def train_model(self):
        """Train the face recognition model"""
        try:
            if len(self.training_data) == 0:
                return False
            
            # Convert labels to numerical format
            encoded_labels = self.label_encoder.fit_transform(self.training_labels)
            
            # Train the recognizer
            self.face_recognizer.train(self.training_data, encoded_labels)
            self.is_trained = True
            
            print(f"✅ Model trained with {len(set(self.training_labels))} persons and {len(self.training_data)} samples")
            return True
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def recognize_face_advanced(self, face_region):
        """
        Advanced face recognition with confidence scoring
        """
        if not self.is_trained or len(self.training_data) == 0:
            return "Unknown", 0, (0, 0, 255)  # Red for unknown
        
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_region)
            if processed_face is None:
                return "Unknown", 0, (0, 0, 255)
            
            # Predict using LBPH recognizer
            label, confidence = self.face_recognizer.predict(processed_face)
            
            # Convert confidence to percentage (lower is better in LBPH)
            confidence_percent = max(0, 100 - confidence)
            
            if confidence < self.recognition_threshold and confidence_percent > 40:
                # Confidence is good, return recognized name
                predicted_name = self.label_encoder.inverse_transform([label])[0]
                color = self.get_color_for_name(predicted_name)
                return predicted_name, confidence_percent, color
            else:
                # Confidence is too low
                return "Unknown", confidence_percent, (0, 0, 255)
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0, (0, 0, 255)
    
    def get_color_for_name(self, name):
        """Get consistent color for each registered name"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203) # Pink
        ]
        
        # Use hash of name to get consistent color
        name_hash = hash(name) % len(colors)
        return colors[name_hash]
    
    def get_registered_names(self):
        """Get list of all registered names"""
        return list(set(self.training_labels)) if self.training_labels else []
    
    def get_training_stats(self):
        """Get training statistics"""
        if not self.training_labels:
            return {"total_persons": 0, "total_samples": 0, "samples_per_person": {}}
        
        unique_names = set(self.training_labels)
        stats = {
            "total_persons": len(unique_names),
            "total_samples": len(self.training_labels),
            "samples_per_person": {}
        }
        
        for name in unique_names:
            count = len([l for l in self.training_labels if l == name])
            stats["samples_per_person"][name] = count
        
        return stats
    
    def delete_person(self, name):
        """Delete all samples for a person"""
        try:
            # Keep only samples that don't match the name
            new_data = []
            new_labels = []
            
            for data, label in zip(self.training_data, self.training_labels):
                if label != name:
                    new_data.append(data)
                    new_labels.append(label)
            
            self.training_data = new_data
            self.training_labels = new_labels
            
            # Retrain model if we have data left
            if len(self.training_data) > 0:
                self.train_model()
            else:
                self.is_trained = False
            
            self.save_training_data()
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False

class PerfectCameraManager:
    """
    Advanced camera management with optimal settings
    """
    def __init__(self):
        self.camera = None
        self.camera_index = None
        self.camera_resolution = (640, 480)
        self.camera_fps = 30
        self.find_best_camera()
    
    def find_best_camera(self):
        """Find the best available camera"""
        print("🔍 Searching for best camera...")
        best_camera = None
        best_resolution = (0, 0)
        
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to get maximum resolution
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"📷 Camera {i}: {width}x{height}")
                    
                    if width * height > best_resolution[0] * best_resolution[1]:
                        best_camera = i
                        best_resolution = (width, height)
                    
                    cap.release()
            except Exception as e:
                print(f"Camera {i} error: {e}")
        
        if best_camera is not None:
            self.camera_index = best_camera
            self.camera_resolution = best_resolution
            print(f"✅ Selected Camera {best_camera} at {best_resolution[0]}x{best_resolution[1]}")
            return True
        else:
            print("❌ No cameras found")
            return False
    
    def start_camera_optimized(self):
        """Start camera with optimal settings"""
        if self.camera_index is None:
            return False
        
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Set optimal camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 0.5)
            
            # Verify camera started successfully
            if self.camera.isOpened():
                # Warm up camera
                for _ in range(5):
                    self.camera.read()
                print("✅ Camera started with optimal settings")
                return True
            return False
        except Exception as e:
            print(f"Camera start error: {e}")
            return False
    
    def read_frame_optimized(self):
        """Read frame with error handling"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Enhance frame quality
                frame = self.enhance_frame(frame)
                return True, frame
        return False, None
    
    def enhance_frame(self, frame):
        """Enhance frame quality for better face detection"""
        try:
            # Increase contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Sharpening filter
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return sharpened
        except:
            return frame
    
    def release(self):
        """Release camera properly"""
        if self.camera:
            self.camera.release()
            self.camera = None

class PerfectSecuritySystem:
    def __init__(self, root):
        self.root = root
        self.root.title("PERFECT Face Recognition Security System")
        self.root.geometry("1400x900")
        
        # Security state
        self.security_enabled = False
        self.camera_manager = PerfectCameraManager()
        self.face_recognition = AdvancedFaceRecognition()
        self.camera_available = self.camera_manager.find_best_camera()
        
        self.setup_ui()
        self.show_system_status()
    
    def show_system_status(self):
        """Show detailed system status"""
        stats = self.face_recognition.get_training_stats()
        
        self.add_notification("🎉 PERFECT FACE RECOGNITION SYSTEM READY!")
        self.add_notification(f"✅ Advanced LBPH Algorithm Active")
        self.add_notification(f"✅ Trained with {stats['total_persons']} persons, {stats['total_samples']} samples")
        
        if self.camera_available:
            res = self.camera_manager.camera_resolution
            self.add_notification(f"✅ Camera: {res[0]}x{res[1]} resolution")
        else:
            self.add_notification("⚠️ Camera not available")

    def setup_ui(self):
        # Configure style
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 20, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 12, 'bold'))
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔒 PERFECT FACE RECOGNITION SECURITY", style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=15)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Status grid
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        self.security_status = ttk.Label(status_grid, text="🔴 SECURITY DISABLED", style='Status.TLabel', foreground='red')
        self.security_status.grid(row=0, column=0, padx=20, sticky='w')
        
        stats = self.face_recognition.get_training_stats()
        self.training_status = ttk.Label(status_grid, text=f"👥 {stats['total_persons']} Persons | {stats['total_samples']} Samples", foreground='blue')
        self.training_status.grid(row=0, column=1, padx=20)
        
        self.algorithm_status = ttk.Label(status_grid, text="🤖 LBPH Algorithm Active", foreground='green')
        self.algorithm_status.grid(row=0, column=2, padx=20)
        
        if self.camera_available:
            res = self.camera_manager.camera_resolution
            self.camera_status = ttk.Label(status_grid, text=f"📷 {res[0]}x{res[1]}", foreground='green')
        else:
            self.camera_status = ttk.Label(status_grid, text="📷 Not Available", foreground='red')
        self.camera_status.grid(row=0, column=3, padx=20)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=15)
        
        button_style = {'font': ('Arial', 10), 'width': 15, 'height': 2}
        
        self.security_btn = tk.Button(control_frame, text="ENABLE SECURITY", 
                                    command=self.toggle_security,
                                    bg='#e74c3c', fg='white', **button_style)
        self.security_btn.pack(side=tk.LEFT, padx=5)
        
        if self.camera_available:
            tk.Button(control_frame, text="LIVE RECOGNITION", 
                     command=self.start_live_recognition,
                     bg='#3498db', fg='white', **button_style).pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="TRAIN NEW FACE", 
                     command=self.train_new_face,
                     bg='#2ecc71', fg='white', **button_style).pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="MANAGE DATABASE", 
                     command=self.manage_database,
                     bg='#9b59b6', fg='white', **button_style).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="SYSTEM STATS", 
                 command=self.show_system_stats,
                 bg='#f39c12', fg='white', **button_style).pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed (left)
        left_frame = ttk.LabelFrame(content_frame, text="Live Face Recognition", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera_label = tk.Label(left_frame, text="Click 'LIVE RECOGNITION' to start\n\nAdvanced LBPH Face Recognition Active", 
                                   bg='black', fg='white', font=('Arial', 12), justify=tk.CENTER)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Recognition results
        results_frame = ttk.LabelFrame(right_frame, text="Recognition Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=6, width=50, font=('Arial', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Confidence meter
        confidence_frame = ttk.LabelFrame(right_frame, text="Recognition Confidence", padding=10)
        confidence_frame.pack(fill=tk.X, pady=5)
        
        self.confidence_canvas = tk.Canvas(confidence_frame, height=30, bg='white')
        self.confidence_canvas.pack(fill=tk.X)
        
        self.confidence_label = ttk.Label(confidence_frame, text="No face detected")
        self.confidence_label.pack()
        
        # Notifications
        notif_frame = ttk.LabelFrame(right_frame, text="System Notifications", padding=10)
        notif_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.notif_text = tk.Text(notif_frame, height=8, font=('Arial', 9))
        self.notif_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(notif_frame, command=self.notif_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.notif_text.config(yscrollcommand=scrollbar.set)
    
    def start_live_recognition(self):
        """Start perfect live face recognition"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera not available")
            return
        
        def live_recognition():
            if self.camera_manager.start_camera_optimized():
                self.add_notification("🔍 Starting PERFECT face recognition...")
                self.add_notification("🤖 LBPH algorithm active - High accuracy mode")
                
                frame_count = 0
                start_time = time.time()
                
                while not self.security_enabled:  # Stop when security starts
                    ret, frame = self.camera_manager.read_frame_optimized()
                    if ret:
                        frame_count += 1
                        fps = frame_count / (time.time() - start_time)
                        
                        # Detect faces with advanced algorithm
                        faces = self.face_recognition.detect_faces_advanced(frame)
                        
                        recognition_results = []
                        confidence_values = []
                        
                        for (x, y, w, h) in faces:
                            face_region = frame[y:y+h, x:x+w]
                            
                            # Recognize face with advanced algorithm
                            name, confidence, color = self.face_recognition.recognize_face_advanced(face_region)
                            
                            recognition_results.append((x, y, w, h, name, confidence, color))
                            confidence_values.append(confidence)
                            
                            # Draw advanced face markers
                            self.draw_advanced_face_marker(frame, x, y, w, h, name, confidence, color)
                        
                        # Update displays
                        self.update_camera_display(frame)
                        self.update_recognition_results(recognition_results, fps)
                        self.update_confidence_meter(confidence_values)
                    
                    time.sleep(0.03)  # ~30 FPS
                
                self.camera_manager.release()
                self.add_notification("✅ Live recognition completed")
        
        threading.Thread(target=live_recognition, daemon=True).start()
    
    def draw_advanced_face_marker(self, frame, x, y, w, h, name, confidence, color):
        """Draw advanced visual markers for face recognition"""
        # Main face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Confidence bar
        bar_width = w
        bar_height = 10
        confidence_width = int(bar_width * (confidence / 100))
        
        cv2.rectangle(frame, (x, y-25), (x+bar_width, y-15), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y-25), (x+confidence_width, y-15), color, -1)
        
        # Name and confidence text
        text = f"{name} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Face center point
        center_x, center_y = x + w//2, y + h//2
        cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        # Face outline points
        cv2.circle(frame, (x, y), 3, color, -1)  # Top-left
        cv2.circle(frame, (x+w, y), 3, color, -1)  # Top-right
        cv2.circle(frame, (x, y+h), 3, color, -1)  # Bottom-left
        cv2.circle(frame, (x+w, y+h), 3, color, -1)  # Bottom-right
    
    def update_camera_display(self, frame):
        """Update camera display with high quality"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Maintain aspect ratio
            display_width = 600
            display_height = 400
            pil_image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_recognition_results(self, results, fps):
        """Update recognition results with detailed information"""
        self.results_text.delete(1.0, tk.END)
        
        if results:
            self.results_text.insert(tk.END, f"FPS: {fps:.1f} | Faces: {len(results)}\n\n")
            
            for i, (x, y, w, h, name, confidence, color) in enumerate(results):
                status_icon = "🟢" if name != "Unknown" else "🔴"
                confidence_color = "green" if confidence > 70 else "orange" if confidence > 40 else "red"
                
                self.results_text.insert(tk.END, f"{status_icon} Face {i+1}:\n")
                self.results_text.insert(tk.END, f"   Name: {name}\n")
                self.results_text.insert(tk.END, f"   Confidence: ")
                self.results_text.insert(tk.END, f"{confidence:.1f}%", confidence_color)
                self.results_text.insert(tk.END, f"\n   Position: ({x}, {y})\n")
                self.results_text.insert(tk.END, f"   Size: {w}x{h}\n\n")
        else:
            self.results_text.insert(tk.END, f"FPS: {fps:.1f}\n\n")
            self.results_text.insert(tk.END, "No faces detected\n")
            self.results_text.insert(tk.END, "• Ensure good lighting\n• Face the camera directly\n• Remove obstructions")
    
    def update_confidence_meter(self, confidence_values):
        """Update confidence meter visualization"""
        self.confidence_canvas.delete("all")
        
        if not confidence_values:
            self.confidence_label.config(text="No face detected")
            return
        
        avg_confidence = sum(confidence_values) / len(confidence_values)
        canvas_width = self.confidence_canvas.winfo_width()
        
        if canvas_width < 10:  # Canvas not yet rendered
            return
        
        # Draw confidence bar
        bar_width = int(canvas_width * (avg_confidence / 100))
        
        if avg_confidence > 70:
            color = "green"
        elif avg_confidence > 40:
            color = "orange"
        else:
            color = "red"
        
        self.confidence_canvas.create_rectangle(0, 0, bar_width, 30, fill=color, outline="")
        self.confidence_canvas.create_rectangle(0, 0, canvas_width, 30, outline="black")
        
        self.confidence_label.config(text=f"Average Confidence: {avg_confidence:.1f}%")
    
    def toggle_security(self):
        """Toggle security monitoring"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera required for security")
            return
        
        self.security_enabled = not self.security_enabled
        
        if self.security_enabled:
            self.security_btn.config(text="DISABLE SECURITY", bg='#2ecc71')
            self.security_status.config(text="🟢 SECURITY ACTIVE", foreground='green')
            self.start_security_monitoring()
            self.add_notification("🔒 PERFECT SECURITY MONITORING ACTIVATED")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.security_status.config(text="🔴 SECURITY DISABLED", foreground='red')
            self.camera_manager.release()
            self.add_notification("Security system deactivated")
    
    def start_security_monitoring(self):
        """Start advanced security monitoring"""
        def security_monitor():
            if self.camera_manager.start_camera_optimized():
                self.add_notification("🚨 SECURITY: Advanced monitoring active")
                self.add_notification("🚨 ALERT: Unknown faces will trigger security protocol")
                
                last_alert_time = 0
                alert_cooldown = 10
                
                while self.security_enabled:
                    ret, frame = self.camera_manager.read_frame_optimized()
                    if ret:
                        faces = self.face_recognition.detect_faces_advanced(frame)
                        
                        security_breach = False
                        for (x, y, w, h) in faces:
                            face_region = frame[y:y+h, x:x+w]
                            name, confidence, color = self.face_recognition.recognize_face_advanced(face_region)
                            
                            if name == "Unknown" and confidence < 50:
                                security_breach = True
                                # Draw security alert
                                cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 3)
                                cv2.putText(frame, "SECURITY BREACH!", (x, y-40), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                # Draw normal recognition
                                self.draw_advanced_face_marker(frame, x, y, w, h, name, confidence, color)
                        
                        self.update_camera_display(frame)
                        
                        # Trigger security alert
                        current_time = time.time()
                        if security_breach and (current_time - last_alert_time) > alert_cooldown:
                            self.trigger_security_alert(frame)
                            last_alert_time = current_time
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
        
        threading.Thread(target=security_monitor, daemon=True).start()
    
    def trigger_security_alert(self, frame):
        """Trigger advanced security alert"""
        try:
            os.makedirs('security_alerts', exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_alerts/breach_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            self.add_notification("🚨🚨🚨 SECURITY BREACH DETECTED!")
            self.add_notification(f"📸 Evidence saved: {filename}")
            self.add_notification("⚠️ Unknown person detected in secured area!")
            
            # Sound alert (if available)
            try:
                import winsound
                for _ in range(5):
                    winsound.Beep(1500, 200)
                    time.sleep(0.1)
            except:
                pass
                
        except Exception as e:
            self.add_notification(f"❌ Alert error: {e}")
    
    def train_new_face(self):
        """Train a new face with multiple samples"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera required for training")
            return
        
        name = simpledialog.askstring("Train New Face", "Enter person's name:")
        if not name:
            return
        
        def advanced_training():
            if self.camera_manager.start_camera_optimized():
                training_window = tk.Toplevel(self.root)
                training_window.title(f"Training: {name}")
                training_window.geometry("600x700")
                
                # Training preview
                preview_label = tk.Label(training_window, bg='black')
                preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                training_info = tk.Label(training_window, text="Position your face in the frame\nMultiple samples will be captured", 
                                       font=('Arial', 12), justify=tk.CENTER)
                training_info.pack(pady=5)
                
                samples_captured = 0
                training_complete = False
                
                def capture_training_samples():
                    nonlocal samples_captured, training_complete
                    
                    samples_to_capture = 10  # Capture multiple samples
                    captured_samples = 0
                    
                    for i in range(100):  # Maximum 100 frames for training
                        if training_complete:
                            break
                            
                        ret, frame = self.camera_manager.read_frame_optimized()
                        if ret:
                            faces = self.face_recognition.detect_faces_advanced(frame)
                            
                            if len(faces) == 1:
                                x, y, w, h = faces[0]
                                
                                # Draw training guidance
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame, f"Training: {name}", (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(frame, f"Samples: {samples_captured}/{samples_to_capture}", (x, y+h+20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                # Capture sample every 5 frames
                                if i % 5 == 0 and captured_samples < samples_to_capture:
                                    success, message = self.face_recognition.register_face(frame, name)
                                    if success:
                                        samples_captured += 1
                                        captured_samples += 1
                                        training_info.config(text=f"Training: {name}\nSamples captured: {samples_captured}/{samples_to_capture}")
                            
                            # Update preview
                            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_image)
                            pil_image.thumbnail((400, 400))
                            photo = ImageTk.PhotoImage(pil_image)
                            preview_label.configure(image=photo)
                            preview_label.image = photo
                        
                        time.sleep(0.1)
                    
                    training_complete = True
                    training_window.destroy()
                    self.camera_manager.release()
                    
                    # Update UI
                    stats = self.face_recognition.get_training_stats()
                    self.training_status.config(text=f"👥 {stats['total_persons']} Persons | {stats['total_samples']} Samples")
                    self.add_notification(f"✅ Training complete: {name} ({samples_captured} samples)")
                
                # Start training in separate thread
                training_thread = threading.Thread(target=capture_training_samples, daemon=True)
                training_thread.start()
                
                # Stop training button
                def stop_training():
                    nonlocal training_complete
                    training_complete = True
                    training_window.destroy()
                    self.camera_manager.release()
                
                tk.Button(training_window, text="STOP TRAINING", 
                         command=stop_training, bg='#e74c3c', fg='white').pack(pady=10)
                
                training_window.mainloop()
            else:
                messagebox.showerror("Camera Error", "Could not start camera")
        
        threading.Thread(target=advanced_training, daemon=True).start()
    
    def manage_database(self):
        """Manage face recognition database"""
        stats = self.face_recognition.get_training_stats()
        
        if stats['total_persons'] == 0:
            messagebox.showinfo("Database Empty", "No faces in database")
            return
        
        db_window = tk.Toplevel(self.root)
        db_window.title("Face Database Management")
        db_window.geometry("500x600")
        
        tk.Label(db_window, text="Face Recognition Database", 
                font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Database statistics
        stats_text = f"Persons: {stats['total_persons']}\nTotal Samples: {stats['total_samples']}"
        tk.Label(db_window, text=stats_text, font=('Arial', 12)).pack(pady=5)
        
        # Person list
        list_frame = ttk.Frame(db_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        listbox = tk.Listbox(list_frame, font=('Arial', 12))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for name, count in stats['samples_per_person'].items():
            listbox.insert(tk.END, f"👤 {name} ({count} samples)")
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def delete_person():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                name = list(stats['samples_per_person'].keys())[index]
                
                if messagebox.askyesno("Confirm Delete", f"Delete all data for '{name}'?"):
                    if self.face_recognition.delete_person(name):
                        listbox.delete(selection)
                        stats = self.face_recognition.get_training_stats()
                        self.training_status.config(text=f"👥 {stats['total_persons']} Persons | {stats['total_samples']} Samples")
                        self.add_notification(f"🗑️ Deleted: {name}")
                    else:
                        messagebox.showerror("Error", "Could not delete person")
        
        tk.Button(db_window, text="Delete Selected", 
                 command=delete_person, bg='#e74c3c', fg='white').pack(pady=10)
    
    def show_system_stats(self):
        """Show detailed system statistics"""
        stats = self.face_recognition.get_training_stats()
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("System Statistics")
        stats_window.geometry("400x500")
        
        tk.Label(stats_window, text="🤖 System Statistics", 
                font=('Arial', 16, 'bold')).pack(pady=10)
        
        stats_text = f"""Algorithm: LBPH Face Recognition
Recognition Threshold: {self.face_recognition.recognition_threshold}
Trained Persons: {stats['total_persons']}
Total Samples: {stats['total_samples']}
Model Status: {'TRAINED' if self.face_recognition.is_trained else 'NOT TRAINED'}
Camera: {'AVAILABLE' if self.camera_available else 'NOT AVAILABLE'}

Persons in Database:"""
        
        text_widget = tk.Text(stats_window, wrap=tk.WORD, height=20)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert("1.0", stats_text)
        
        for name, count in stats['samples_per_person'].items():
            text_widget.insert(tk.END, f"\n• {name}: {count} samples")
        
        text_widget.config(state=tk.DISABLED)
    
    def add_notification(self, message):
        """Add notification with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.notif_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.notif_text.see(tk.END)

def main():
    try:
        root = tk.Tk()
        app = PerfectSecuritySystem(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Failed to start: {e}")

if __name__ == "__main__":
    main()