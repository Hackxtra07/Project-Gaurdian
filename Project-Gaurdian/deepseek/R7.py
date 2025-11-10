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
import subprocess
import sys

print("Loading security system...")

# Enhanced import with better error handling
FACE_RECOGNITION_AVAILABLE = False
face_recognition = None

try:
    # Try to import with explicit error handling
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition library imported successfully!")
    
    # Test if models are available
    try:
        # This will trigger the model loading and show if models are missing
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_recognition.face_locations(test_image)
        print("✅ face_recognition models are available!")
    except Exception as model_error:
        print(f"❌ Face recognition models error: {model_error}")
        FACE_RECOGNITION_AVAILABLE = False
        raise ImportError("Models not available")
        
except ImportError as e:
    print(f"❌ Face recognition not available: {e}")
    FACE_RECOGNITION_AVAILABLE = False

except Exception as e:
    print(f"❌ Unexpected error with face_recognition: {e}")
    FACE_RECOGNITION_AVAILABLE = False

class FaceRecognitionSystem:
    def __init__(self):
        if FACE_RECOGNITION_AVAILABLE:
            self.known_face_encodings = []
            self.known_face_names = []
        else:
            self.known_face_names = []
            self.face_colors = {}
        
        self.load_face_data()
    
    def load_face_data(self):
        """Load face data based on available system"""
        try:
            if FACE_RECOGNITION_AVAILABLE:
                if os.path.exists('real_face_data.pkl'):
                    with open('real_face_data.pkl', 'rb') as f:
                        data = pickle.load(f)
                        self.known_face_encodings = data['encodings']
                        self.known_face_names = data['names']
                    print(f"✅ Loaded {len(self.known_face_names)} known faces")
            else:
                if os.path.exists('simulated_faces.pkl'):
                    with open('simulated_faces.pkl', 'rb') as f:
                        data = pickle.load(f)
                        self.known_face_names = data.get('names', [])
                        self.face_colors = data.get('colors', {})
                else:
                    # Create default faces for simulation
                    self.known_face_names = ['Admin', 'User']
                    self.face_colors = {'Admin': (0, 255, 0), 'User': (0, 200, 100)}
                    self.save_face_data()
                    
        except Exception as e:
            print(f"Error loading face data: {e}")
    
    def save_face_data(self):
        """Save face data"""
        try:
            if FACE_RECOGNITION_AVAILABLE:
                data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }
                with open('real_face_data.pkl', 'wb') as f:
                    pickle.dump(data, f)
            else:
                data = {
                    'names': self.known_face_names,
                    'colors': self.face_colors
                }
                with open('simulated_faces.pkl', 'wb') as f:
                    pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving face data: {e}")
            return False
    
    def register_face(self, image, name):
        """Register a new face"""
        if FACE_RECOGNITION_AVAILABLE:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                if len(face_locations) == 0:
                    return False, "No face detected"
                elif len(face_locations) > 1:
                    return False, "Multiple faces detected"
                else:
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    if len(face_encodings) == 1:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        self.save_face_data()
                        return True, f"Face registered: {name}"
                    else:
                        return False, "Could not extract face features"
            except Exception as e:
                return False, f"Registration error: {str(e)}"
        else:
            # Simulated registration
            if name not in self.known_face_names:
                self.known_face_names.append(name)
                # Assign a random color for this face
                colors = [(0, 255, 0), (0, 200, 100), (100, 200, 0), (0, 100, 255)]
                self.face_colors[name] = colors[len(self.known_face_names) % len(colors)]
                self.save_face_data()
                return True, f"Simulated face registered: {name}"
            else:
                return False, "Name already exists"
    
    def recognize_faces(self, image, face_locations=None):
        """Recognize faces in image"""
        if FACE_RECOGNITION_AVAILABLE:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if face_locations is None:
                    face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                            name = self.known_face_names[best_match_index]
                    
                    face_names.append(name)
                
                return face_locations, face_names
            except Exception as e:
                print(f"Recognition error: {e}")
                return [], []
        else:
            # Simulated recognition
            if not face_locations:
                return [], []
            
            face_names = []
            for _ in face_locations:
                if self.known_face_names and np.random.random() > 0.3:  # 70% chance of recognizing
                    name = np.random.choice(self.known_face_names)
                else:
                    name = "Unknown"
                face_names.append(name)
            
            return face_locations, face_names
    
    def get_face_color(self, name):
        """Get color for face box"""
        if FACE_RECOGNITION_AVAILABLE:
            return (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        else:
            return self.face_colors.get(name, (0, 0, 255))

class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_index = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.find_camera()
    
    def find_camera(self):
        """Find working camera"""
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"✅ Camera {i} found")
                        self.camera_index = i
                        cap.release()
                        return True
                    cap.release()
            except:
                pass
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
            return self.camera.isOpened()
        except:
            return False
    
    def detect_faces_opencv(self, frame):
        """Detect faces using OpenCV (fallback)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        # Convert to face_recognition format (top, right, bottom, left)
        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
        return face_locations
    
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
        self.root.title("Security System - Universal Face Recognition")
        self.root.geometry("1200x800")
        
        self.security_enabled = False
        self.camera_manager = CameraManager()
        self.face_system = FaceRecognitionSystem()
        self.camera_available = self.camera_manager.find_camera()
        
        self.setup_ui()
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Show welcome message based on system capabilities"""
        if FACE_RECOGNITION_AVAILABLE:
            self.add_notification("🎉 REAL Face Recognition System Ready!")
            self.add_notification(f"✅ Loaded {len(self.face_system.known_face_names)} known faces")
        else:
            self.add_notification("🔧 Running in SIMULATED Face Recognition Mode")
            self.add_notification("💡 Install face_recognition for real face recognition")
        
        if self.camera_available:
            self.add_notification("✅ Camera detected and ready")
        else:
            self.add_notification("⚠️ Camera not available - using demo mode")
        
        # Show installation help if face_recognition is not available
        if not FACE_RECOGNITION_AVAILABLE:
            self.add_notification("💻 Run: pip install face_recognition")

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title with mode indicator
        mode = "REAL" if FACE_RECOGNITION_AVAILABLE else "SIMULATED"
        title_label = tk.Label(main_frame, text=f"🔒 SECURITY SYSTEM - {mode} FACE RECOGNITION", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔴 SECURITY DISABLED", 
                                   font=('Arial', 12, 'bold'), fg='red', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT)
        
        # Mode indicator
        mode_color = 'green' if FACE_RECOGNITION_AVAILABLE else 'orange'
        mode_text = "REAL FACE RECOGNITION" if FACE_RECOGNITION_AVAILABLE else "SIMULATED MODE"
        self.mode_label = tk.Label(status_frame, text=mode_text, 
                                 font=('Arial', 10), fg=mode_color, bg='#2c3e50')
        self.mode_label.pack(side=tk.LEFT, padx=20)
        
        # Faces count
        face_count = len(self.face_system.known_face_names)
        self.faces_label = tk.Label(status_frame, text=f"Known Faces: {face_count}", 
                                  font=('Arial', 10), fg='blue', bg='#2c3e50')
        self.faces_label.pack(side=tk.LEFT, padx=20)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.security_btn = tk.Button(control_frame, text="ENABLE SECURITY", 
                                    command=self.toggle_security,
                                    font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                    width=15, height=2)
        self.security_btn.pack(side=tk.LEFT, padx=5)
        
        if self.camera_available:
            tk.Button(control_frame, text="TEST CAMERA", 
                     command=self.test_camera,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="REGISTER FACE", 
                     command=self.register_face,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="MANAGE FACES", 
                 command=self.manage_faces,
                 font=('Arial', 10), bg='#9b59b6', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="BACKUP FILES", 
                 command=self.backup_files,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed
        left_frame = ttk.LabelFrame(content_frame, text="Live Camera Feed", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        if self.camera_available:
            self.camera_label = tk.Label(left_frame, text="Camera Ready - Click TEST CAMERA", 
                                       bg='black', fg='white', font=('Arial', 14))
        else:
            self.camera_label = tk.Label(left_frame, text="DEMO MODE - No Camera Available", 
                                       bg='black', fg='white', font=('Arial', 14))
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Recognition results
        results_frame = ttk.LabelFrame(right_frame, text="Recognition Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=4, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Notifications
        notif_frame = ttk.LabelFrame(right_frame, text="Notifications", padding=10)
        notif_frame.pack(fill=tk.X, pady=5)
        
        self.notif_text = tk.Text(notif_frame, height=8, width=40)
        self.notif_text.pack(fill=tk.BOTH, expand=True)
        
        # Logs
        log_frame = ttk.LabelFrame(right_frame, text="Security Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.log_text = tk.Text(log_frame, height=6)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def test_camera(self):
        """Test camera with face detection"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera not available")
            return
        
        def test_sequence():
            if self.camera_manager.start_camera():
                self.add_notification("🔍 Testing camera...")
                for _ in range(50):
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Detect faces
                        if FACE_RECOGNITION_AVAILABLE:
                            face_locations = face_recognition.face_locations(frame)
                        else:
                            face_locations = self.camera_manager.detect_faces_opencv(frame)
                        
                        # Recognize faces
                        face_locations, face_names = self.face_system.recognize_faces(frame, face_locations)
                        
                        # Draw results
                        frame = self.draw_face_boxes(frame, face_locations, face_names)
                        self.update_display(frame)
                        self.update_results(face_names)
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
                self.add_notification("✅ Camera test completed")
        
        threading.Thread(target=test_sequence, daemon=True).start()
    
    def draw_face_boxes(self, frame, face_locations, face_names):
        """Draw face recognition boxes"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            color = self.face_system.get_face_color(name)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        return frame
    
    def update_display(self, frame):
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
    
    def update_results(self, face_names):
        """Update recognition results"""
        self.results_text.delete(1.0, tk.END)
        if face_names:
            for i, name in enumerate(face_names):
                status = "✅ KNOWN" if name != "Unknown" else "🚨 UNKNOWN"
                self.results_text.insert(tk.END, f"Face {i+1}: {name} {status}\n")
        else:
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
            self.start_monitoring()
            mode = "REAL" if FACE_RECOGNITION_AVAILABLE else "SIMULATED"
            self.add_notification(f"🔒 Security monitoring started ({mode} mode)")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.status_label.config(text="🔴 SECURITY DISABLED", fg='red')
            self.camera_manager.release()
            self.add_notification("Security system deactivated")
    
    def start_monitoring(self):
        """Start security monitoring"""
        def monitor():
            if self.camera_manager.start_camera():
                last_alert = 0
                
                while self.security_enabled:
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Detect faces
                        if FACE_RECOGNITION_AVAILABLE:
                            face_locations = face_recognition.face_locations(frame)
                        else:
                            face_locations = self.camera_manager.detect_faces_opencv(frame)
                        
                        # Recognize faces
                        face_locations, face_names = self.face_system.recognize_faces(frame, face_locations)
                        
                        # Update display
                        frame = self.draw_face_boxes(frame, face_locations, face_names)
                        self.update_display(frame)
                        self.update_results(face_names)
                        
                        # Security alerts
                        current_time = time.time()
                        if "Unknown" in face_names and (current_time - last_alert) > 10:
                            self.trigger_alert(frame)
                            last_alert = current_time
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def trigger_alert(self, frame):
        """Trigger security alert"""
        try:
            os.makedirs('captured_images', exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_images/alert_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.add_notification("🚨 ALERT! Unknown face detected!")
            self.log_event(f"ALERT: Unknown face - {filename}")
        except Exception as e:
            self.log_event(f"Alert error: {e}")
    
    def register_face(self):
        """Register new face"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera required for face registration")
            return
        
        name = simpledialog.askstring("Register Face", "Enter name:")
        if not name:
            return
        
        def capture_face():
            if self.camera_manager.start_camera():
                time.sleep(1)  # Let camera stabilize
                ret, frame = self.camera_manager.read_frame()
                if ret:
                    success, message = self.face_system.register_face(frame, name)
                    if success:
                        self.add_notification(f"✅ {message}")
                        self.faces_label.config(text=f"Known Faces: {len(self.face_system.known_face_names)}")
                    else:
                        self.add_notification(f"❌ {message}")
                    messagebox.showinfo("Result", message)
                else:
                    messagebox.showerror("Error", "Could not capture image")
                self.camera_manager.release()
        
        threading.Thread(target=capture_face, daemon=True).start()
    
    def manage_faces(self):
        """Manage registered faces"""
        if not self.face_system.known_face_names:
            messagebox.showinfo("No Faces", "No faces registered yet")
            return
        
        window = tk.Toplevel(self.root)
        window.title("Manage Faces")
        window.geometry("300x400")
        
        tk.Label(window, text="Registered Faces", font=('Arial', 14, 'bold')).pack(pady=10)
        
        listbox = tk.Listbox(window, font=('Arial', 12))
        for name in self.face_system.known_face_names:
            listbox.insert(tk.END, f"👤 {name}")
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def delete_face():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                name = self.face_system.known_face_names[index]
                if messagebox.askyesno("Confirm", f"Delete {name}?"):
                    if FACE_RECOGNITION_AVAILABLE:
                        del self.face_system.known_face_names[index]
                        del self.face_system.known_face_encodings[index]
                    else:
                        del self.face_system.known_face_names[index]
                        if name in self.face_system.face_colors:
                            del self.face_system.face_colors[name]
                    
                    self.face_system.save_face_data()
                    listbox.delete(selection)
                    self.faces_label.config(text=f"Known Faces: {len(self.face_system.known_face_names)}")
                    self.add_notification(f"🗑️ Deleted: {name}")
        
        tk.Button(window, text="Delete Selected", command=delete_face, 
                 bg='#e74c3c', fg='white').pack(pady=10)
    
    def backup_files(self):
        """Backup files"""
        try:
            source_dir = filedialog.askdirectory(title="Select folder to backup")
            if source_dir:
                backup_name = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copytree(source_dir, backup_name)
                self.add_notification(f"✅ Backup: {backup_name}")
                messagebox.showinfo("Success", f"Backup created: {backup_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Backup failed: {e}")
    
    def add_notification(self, message):
        """Add notification"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.notif_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.notif_text.see(tk.END)
    
    def log_event(self, event):
        """Log security event"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {event}\n")
        self.log_text.see(tk.END)

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