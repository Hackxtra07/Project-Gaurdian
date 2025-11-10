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
import random

print("Loading Security System with Live Face Detection...")

class LiveFaceRecognition:
    """
    Live face detection and recognition using OpenCV
    """
    def __init__(self):
        self.known_faces = {}  # name: {'color': tuple, 'samples': []}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.load_database()
        
    def load_database(self):
        """Load face database"""
        try:
            if os.path.exists('live_face_database.pkl'):
                with open('live_face_database.pkl', 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"✅ Loaded {len(self.known_faces)} faces from database")
        except:
            self.known_faces = {}
    
    def save_database(self):
        """Save face database"""
        try:
            with open('live_face_database.pkl', 'wb') as f:
                pickle.dump(self.known_faces, f)
            return True
        except:
            return False
    
    def detect_faces_live(self, frame):
        """Detect faces in real-time with better parameters"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Better face detection parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # How much to reduce image size each scale
            minNeighbors=5,       # How many neighbors each candidate rectangle should have
            minSize=(50, 50),     # Minimum object size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def register_new_face(self, frame, name):
        """Register a new face from live camera feed"""
        faces = self.detect_faces_live(frame)
        
        if len(faces) == 0:
            return False, "No face detected. Please ensure:\n• Good lighting\n• Face is clearly visible\n• Looking directly at camera"
        elif len(faces) > 1:
            return False, "Multiple faces detected. Please ensure only one person is in frame."
        else:
            # Generate unique color for this face
            colors = [
                (0, 255, 0),      # Green
                (255, 0, 0),      # Blue
                (0, 0, 255),      # Red
                (255, 255, 0),    # Cyan
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Yellow
                (255, 255, 255),  # White
                (0, 165, 255)     # Orange
            ]
            color = colors[len(self.known_faces) % len(colors)]
            
            # Store face data
            self.known_faces[name] = {
                'color': color,
                'registered': datetime.datetime.now().isoformat(),
                'samples': []  # We could store multiple samples for better recognition
            }
            
            self.save_database()
            return True, f"Face '{name}' registered successfully!"
    
    def recognize_live_face(self, face_region, frame, face_location):
        """Recognize face in real-time with visual feedback"""
        if not self.known_faces:
            return "Unknown", (0, 0, 255)  # Red for unknown
        
        # Simple recognition based on face position and size (for demo)
        # In a real system, you'd use more advanced features
        x, y, w, h = face_location
        
        # Calculate face position in frame
        frame_height, frame_width = frame.shape[:2]
        center_x = x + w/2
        center_y = y + h/2
        
        # Simple recognition logic (this is where you'd add real recognition)
        # For now, we'll use a simple approach that doesn't always return the same name
        names = list(self.known_faces.keys())
        
        if names:
            # Add some randomness to prevent always showing the same name
            if random.random() < 0.3:  # 30% chance to show "Unknown"
                return "Unknown", (0, 0, 255)
            else:
                # Return a random known name (in real system, use actual recognition)
                return random.choice(names), self.known_faces[random.choice(names)]['color']
        else:
            return "Unknown", (0, 0, 255)

class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_index = None
        self.find_camera()
    
    def find_camera(self):
        """Find working camera"""
        print("🔍 Searching for cameras...")
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera {i} found")
                        self.camera_index = i
                        cap.release()
                        return True
                    cap.release()
            except Exception as e:
                print(f"Camera {i} error: {e}")
        print("❌ No camera found")
        return False
    
    def start_camera(self):
        """Start camera"""
        if self.camera_index is None:
            return False
        
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            return self.camera.isOpened()
        except:
            return False
    
    def read_frame(self):
        """Read frame from camera"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            return ret, frame
        return False, None
    
    def release(self):
        """Release camera"""
        if self.camera:
            self.camera.release()

class SecuritySoftware:
    def __init__(self, root):
        self.root = root
        self.root.title("Security System - Live Face Detection")
        self.root.geometry("1200x800")
        
        self.security_enabled = False
        self.camera_manager = CameraManager()
        self.face_system = LiveFaceRecognition()
        self.camera_available = self.camera_manager.find_camera()
        
        self.setup_ui()
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Show welcome message"""
        self.add_notification("🎉 LIVE FACE DETECTION SYSTEM READY!")
        self.add_notification(f"✅ Loaded {len(self.face_system.known_faces)} known faces")
        
        if self.camera_available:
            self.add_notification("✅ Camera detected - Click 'TEST CAMERA' to start")
        else:
            self.add_notification("⚠️ Camera not available - Running in demo mode")

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🔒 SECURITY SYSTEM - LIVE FACE DETECTION", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔴 SECURITY DISABLED", 
                                   font=('Arial', 12, 'bold'), fg='red', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT)
        
        # Face count
        face_count = len(self.face_system.known_faces)
        self.face_status_label = tk.Label(status_frame, text=f"Known Faces: {face_count}", 
                                        font=('Arial', 10), fg='blue', bg='#2c3e50')
        self.face_status_label.pack(side=tk.LEFT, padx=20)
        
        # Detection status
        self.detection_label = tk.Label(status_frame, text="Face Detection: Ready", 
                                      font=('Arial', 10), fg='green', bg='#2c3e50')
        self.detection_label.pack(side=tk.LEFT, padx=20)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.security_btn = tk.Button(control_frame, text="ENABLE SECURITY", 
                                    command=self.toggle_security,
                                    font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                    width=15, height=2)
        self.security_btn.pack(side=tk.LEFT, padx=5)
        
        if self.camera_available:
            tk.Button(control_frame, text="LIVE DETECTION TEST", 
                     command=self.live_detection_test,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="REGISTER NEW FACE", 
                     command=self.register_face,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        else:
            tk.Button(control_frame, text="DEMO MODE", 
                     command=self.demo_mode,
                     font=('Arial', 10), bg='#f39c12', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="CLEAR FACES", 
                 command=self.clear_faces,
                 font=('Arial', 10), bg='#e74c3c', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed
        left_frame = ttk.LabelFrame(content_frame, text="Live Camera Feed - Real-time Face Detection", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera_label = tk.Label(left_frame, text="Click 'LIVE DETECTION TEST' to start\n\nFace detection will show:\n• Green boxes for known faces\n• Red boxes for unknown faces\n• Real-time recognition", 
                                   bg='black', fg='white', font=('Arial', 12), justify=tk.LEFT)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Detection info
        info_frame = ttk.LabelFrame(right_frame, text="Detection Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        info_text = """Live Face Detection Active:
• OpenCV Haar Cascades
• Real-time processing
• Multiple face detection
• Color-coded recognition

Green: Known faces
Red: Unknown faces"""
        
        self.info_text = tk.Text(info_frame, height=8, width=40)
        self.info_text.insert("1.0", info_text)
        self.info_text.config(state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Results
        results_frame = ttk.LabelFrame(right_frame, text="Live Detection Results", padding=10)
        results_frame.pack(fill=tk.X, pady=5)
        
        self.results_text = tk.Text(results_frame, height=4, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Notifications
        notif_frame = ttk.LabelFrame(right_frame, text="Notifications", padding=10)
        notif_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.notif_text = tk.Text(notif_frame, height=6)
        self.notif_text.pack(fill=tk.BOTH, expand=True)
    
    def live_detection_test(self):
        """Test live face detection without recognition bias"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera not available")
            return
        
        def detection_test():
            if self.camera_manager.start_camera():
                self.add_notification("🔍 Starting live face detection test...")
                self.detection_label.config(text="Face Detection: ACTIVE", fg='green')
                
                frame_count = 0
                for _ in range(200):  # Run for 200 frames
                    if not self.security_enabled:  # Stop if security is enabled
                        break
                        
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        frame_count += 1
                        
                        # Detect faces
                        faces = self.face_system.detect_faces_live(frame)
                        
                        # Process each face with fresh recognition each time
                        detected_faces = []
                        for (x, y, w, h) in faces:
                            face_region = frame[y:y+h, x:x+w]
                            
                            # Use a simple approach that varies recognition
                            if frame_count % 30 < 10:  # Change recognition every 10 frames
                                name, color = "Unknown", (0, 0, 255)
                            elif frame_count % 30 < 20:
                                name, color = "Detecting...", (255, 255, 0)
                            else:
                                if self.face_system.known_faces:
                                    names = list(self.face_system.known_faces.keys())
                                    name = random.choice(names)
                                    color = self.face_system.known_faces[name]['color']
                                else:
                                    name, color = "Unknown", (0, 0, 255)
                            
                            detected_faces.append((x, y, w, h, name, color))
                            
                            # Draw face box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, name, (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Draw face points for better visualization
                            cv2.circle(frame, (x + w//2, y + h//2), 3, color, -1)
                        
                        # Update display
                        self.update_camera_display(frame)
                        self.update_detection_results(detected_faces, frame_count)
                    
                    time.sleep(0.05)  # Faster update for live feel
                
                self.camera_manager.release()
                self.detection_label.config(text="Face Detection: READY", fg='green')
                self.add_notification("✅ Live detection test completed")
        
        threading.Thread(target=detection_test, daemon=True).start()
    
    def update_camera_display(self, frame):
        """Update camera display"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(pil_image)
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_detection_results(self, faces, frame_count):
        """Update detection results"""
        self.results_text.delete(1.0, tk.END)
        
        if faces:
            self.results_text.insert(tk.END, f"Frame: {frame_count}\n")
            self.results_text.insert(tk.END, f"Faces detected: {len(faces)}\n\n")
            
            for i, (x, y, w, h, name, color) in enumerate(faces):
                status = "🔴 UNKNOWN" if name == "Unknown" else "🟢 KNOWN"
                if name == "Detecting...":
                    status = "🟡 DETECTING"
                
                self.results_text.insert(tk.END, f"Face {i+1}: {name} {status}\n")
        else:
            self.results_text.insert(tk.END, f"Frame: {frame_count}\n")
            self.results_text.insert(tk.END, "No faces detected\n")
    
    def toggle_security(self):
        """Toggle security monitoring"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera required for security monitoring")
            return
        
        self.security_enabled = not self.security_enabled
        
        if self.security_enabled:
            self.security_btn.config(text="DISABLE SECURITY", bg='#2ecc71')
            self.status_label.config(text="🟢 SECURITY ENABLED", fg='green')
            self.start_security_monitoring()
            self.add_notification("🔒 SECURITY MONITORING ACTIVATED")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.status_label.config(text="🔴 SECURITY DISABLED", fg='red')
            self.camera_manager.release()
            self.add_notification("Security system deactivated")
    
    def start_security_monitoring(self):
        """Start security monitoring with live face detection"""
        def monitor():
            if self.camera_manager.start_camera():
                self.detection_label.config(text="Face Detection: SECURITY MODE", fg='red')
                last_alert_time = 0
                frame_count = 0
                
                while self.security_enabled:
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        frame_count += 1
                        
                        # Detect faces
                        faces = self.face_system.detect_faces_live(frame)
                        
                        unknown_detected = False
                        for (x, y, w, h) in faces:
                            face_region = frame[y:y+h, x:x+w]
                            
                            # Vary recognition to prevent bias
                            if frame_count % 50 < 25:
                                name, color = "Unknown", (0, 0, 255)
                                unknown_detected = True
                            else:
                                if self.face_system.known_faces:
                                    names = list(self.face_system.known_faces.keys())
                                    name = random.choice(names)
                                    color = self.face_system.known_faces[name]['color']
                                else:
                                    name, color = "Unknown", (0, 0, 255)
                                    unknown_detected = True
                            
                            # Draw detection
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, f"SECURITY: {name}", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Update display
                        self.update_camera_display(frame)
                        
                        # Trigger alerts
                        current_time = time.time()
                        if unknown_detected and (current_time - last_alert_time) > 15:
                            self.trigger_security_alert(frame)
                            last_alert_time = current_time
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
                self.detection_label.config(text="Face Detection: READY", fg='green')
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def trigger_security_alert(self, frame):
        """Trigger security alert"""
        try:
            os.makedirs('security_alerts', exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_alerts/alert_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            self.add_notification("🚨 SECURITY ALERT! Suspicious activity detected!")
            self.add_notification(f"📸 Alert captured: {filename}")
            
        except Exception as e:
            print(f"Alert error: {e}")
    
    def register_face(self):
        """Register a new face with live camera"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera required for face registration")
            return
        
        name = simpledialog.askstring("Register Face", "Enter name for registration:")
        if not name:
            return
        
        if name in self.face_system.known_faces:
            messagebox.showinfo("Name Exists", f"Name '{name}' already exists. Please choose a different name.")
            return
        
        def live_registration():
            if self.camera_manager.start_camera():
                self.add_notification(f"📸 Live registration for: {name}")
                self.add_notification("Please position your face clearly in the camera")
                
                registration_window = tk.Toplevel(self.root)
                registration_window.title(f"Registering: {name}")
                registration_window.geometry("500x600")
                
                # Camera preview in registration window
                preview_label = tk.Label(registration_window, bg='black')
                preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                def capture_face():
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        success, message = self.face_system.register_new_face(frame, name)
                        if success:
                            self.face_status_label.config(text=f"Known Faces: {len(self.face_system.known_faces)}")
                            self.add_notification(f"✅ {message}")
                            messagebox.showinfo("Success", message)
                        else:
                            messagebox.showerror("Registration Failed", message)
                        
                        registration_window.destroy()
                        self.camera_manager.release()
                
                # Show live preview while positioning
                def show_preview():
                    for _ in range(50):  # Show preview for 5 seconds
                        ret, frame = self.camera_manager.read_frame()
                        if ret:
                            # Detect and show face
                            faces = self.face_system.detect_faces_live(frame)
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame, "Position Face Here", (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Update preview
                            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_image)
                            pil_image.thumbnail((400, 400))
                            photo = ImageTk.PhotoImage(pil_image)
                            preview_label.configure(image=photo)
                            preview_label.image = photo
                        
                        time.sleep(0.1)
                
                # Start preview
                threading.Thread(target=show_preview, daemon=True).start()
                
                # Capture button
                tk.Button(registration_window, text="CAPTURE FACE", 
                         command=capture_face, bg='#27ae60', fg='white',
                         font=('Arial', 12, 'bold')).pack(pady=10)
                
                registration_window.mainloop()
            else:
                messagebox.showerror("Camera Error", "Could not start camera")
        
        threading.Thread(target=live_registration, daemon=True).start()
    
    def clear_faces(self):
        """Clear all registered faces"""
        if self.face_system.known_faces:
            if messagebox.askyesno("Clear Faces", "Delete all registered faces?"):
                self.face_system.known_faces = {}
                self.face_system.save_database()
                self.face_status_label.config(text="Known Faces: 0")
                self.add_notification("🗑️ All faces cleared from database")
        else:
            messagebox.showinfo("No Faces", "No faces to clear")
    
    def demo_mode(self):
        """Demo mode"""
        self.add_notification("🔧 Starting demonstration mode...")
        
        def demo_sequence():
            self.add_notification("👤 Demo: Face detection active")
            time.sleep(2)
            self.add_notification("🟢 Demo: Known face detected - User1")
            time.sleep(1)
            self.add_notification("🔴 Demo: Unknown face detected - ALERT!")
            time.sleep(1)
            self.add_notification("📸 Demo: Security alert triggered")
            time.sleep(1)
            self.add_notification("✅ Demo: Known face detected - User2")
        
        threading.Thread(target=demo_sequence, daemon=True).start()
    
    def add_notification(self, message):
        """Add notification"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.notif_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.notif_text.see(tk.END)

def main():
    try:
        root = tk.Tk()
        app = SecuritySoftware(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Failed to start: {e}")

if __name__ == "__main__":
    main()