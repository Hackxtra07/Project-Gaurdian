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

print("Loading Security System with OpenCV Face Detection...")

class FaceDatabase:
    """
    Simple face database using OpenCV face detection and basic feature matching
    """
    def __init__(self):
        self.known_faces = {}  # name: {'features': [], 'color': tuple, 'registered': timestamp}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_database()
    
    def load_database(self):
        """Load face database from file"""
        try:
            if os.path.exists('face_database.pkl'):
                with open('face_database.pkl', 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"✅ Loaded {len(self.known_faces)} faces from database")
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            self.known_faces = {}
    
    def save_database(self):
        """Save face database to file"""
        try:
            with open('face_database.pkl', 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"✅ Saved {len(self.known_faces)} faces to database")
            return True
        except Exception as e:
            print(f"❌ Error saving database: {e}")
            return False
    
    def extract_face_features(self, face_region):
        """
        Extract simple features from face region for basic matching
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            standardized = cv2.resize(gray, (100, 100))
            
            # Simple features: histogram and basic statistics
            hist = cv2.calcHist([standardized], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Add some basic shape features
            mean_intensity = np.mean(standardized)
            std_intensity = np.std(standardized)
            
            features = np.concatenate([hist, [mean_intensity, std_intensity]])
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        try:
            if features1 is None or features2 is None:
                return 0
            # Use cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
        except:
            return 0
    
    def register_face(self, image, name):
        """Register a new face in the database"""
        try:
            # Detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            if len(faces) == 0:
                return False, "No face detected. Please ensure good lighting and clear face visibility."
            elif len(faces) > 1:
                return False, "Multiple faces detected. Please ensure only one face is visible."
            else:
                x, y, w, h = faces[0]
                face_region = image[y:y+h, x:x+w]
                
                # Extract features
                features = self.extract_face_features(face_region)
                if features is None:
                    return False, "Could not extract face features. Please try again."
                
                # Check if name already exists
                if name in self.known_faces:
                    return False, f"Name '{name}' already exists. Please use a different name."
                
                # Generate a unique color for this face
                colors = [
                    (0, 255, 0),    # Green
                    (255, 255, 0),  # Cyan
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Yellow
                    (255, 255, 255) # White
                ]
                color = colors[len(self.known_faces) % len(colors)]
                
                # Save to database
                self.known_faces[name] = {
                    'features': features,
                    'color': color,
                    'registered': datetime.datetime.now().isoformat(),
                    'face_region': face_region  # Store for display purposes
                }
                
                self.save_database()
                return True, f"Face '{name}' registered successfully!"
                
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def recognize_face(self, face_region):
        """Recognize a face from the database"""
        try:
            if not self.known_faces:
                return "Unknown", (0, 0, 255), 0.0  # Red for unknown
            
            # Extract features from current face
            current_features = self.extract_face_features(face_region)
            if current_features is None:
                return "Unknown", (0, 0, 255), 0.0
            
            best_match = None
            best_similarity = 0
            threshold = 0.6  # Similarity threshold for recognition
            
            for name, data in self.known_faces.items():
                similarity = self.calculate_similarity(current_features, data['features'])
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = name
            
            if best_match:
                return best_match, self.known_faces[best_match]['color'], best_similarity
            else:
                return "Unknown", (0, 0, 255), 0.0
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", (0, 0, 255), 0.0

class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_index = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.find_camera()
    
    def find_camera(self):
        """Find a working camera"""
        print("🔍 Searching for cameras...")
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera {i} found and working")
                        self.camera_index = i
                        cap.release()
                        return True
                    cap.release()
            except Exception as e:
                print(f"Camera {i} error: {e}")
        print("❌ No working camera found")
        return False
    
    def start_camera(self):
        """Start the camera"""
        if self.camera_index is None:
            if not self.find_camera():
                return False
        
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if self.camera.isOpened():
                print(f"✅ Camera started at index {self.camera_index}")
                return True
            return False
        except Exception as e:
            print(f"Camera start error: {e}")
            return False
    
    def detect_faces(self, image):
        """Detect faces in image using OpenCV"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
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
            self.camera = None

class SecuritySoftware:
    def __init__(self, root):
        self.root = root
        self.root.title("Security System - OpenCV Face Recognition")
        self.root.geometry("1300x850")
        self.root.configure(bg='#2c3e50')
        
        # Security state
        self.security_enabled = False
        self.camera_manager = CameraManager()
        self.face_database = FaceDatabase()
        self.camera_available = self.camera_manager.find_camera()
        
        self.setup_ui()
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Show welcome message"""
        self.add_notification("🎉 SECURITY SYSTEM READY!")
        self.add_notification("✅ OpenCV Face Detection & Recognition Active")
        self.add_notification(f"✅ Loaded {len(self.face_database.known_faces)} known faces")
        
        if self.camera_available:
            self.add_notification("✅ Camera detected and ready")
        else:
            self.add_notification("⚠️ Camera not available - using demo mode")
        
        self.add_notification("💡 Click 'TEST CAMERA' to start")

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🔒 SECURITY SYSTEM - OPENCV FACE RECOGNITION", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔴 SECURITY DISABLED", 
                                   font=('Arial', 12, 'bold'), fg='red', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT)
        
        # Face database status
        face_count = len(self.face_database.known_faces)
        self.face_status_label = tk.Label(status_frame, text=f"Known Faces: {face_count}", 
                                        font=('Arial', 10), fg='blue', bg='#2c3e50')
        self.face_status_label.pack(side=tk.LEFT, padx=20)
        
        # Recognition method
        self.method_label = tk.Label(status_frame, text="Method: OpenCV + Feature Matching", 
                                   font=('Arial', 10), fg='green', bg='#2c3e50')
        self.method_label.pack(side=tk.LEFT, padx=20)
        
        # Camera status
        camera_status = "✅ Available" if self.camera_available else "❌ Not Available"
        self.camera_status_label = tk.Label(status_frame, text=f"Camera: {camera_status}", 
                                          font=('Arial', 10), fg='green' if self.camera_available else 'red', bg='#2c3e50')
        self.camera_status_label.pack(side=tk.RIGHT)
        
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
            
            tk.Button(control_frame, text="VIEW FACES", 
                     command=self.view_faces,
                     font=('Arial', 10), bg='#9b59b6', fg='white').pack(side=tk.LEFT, padx=5)
        else:
            tk.Button(control_frame, text="DEMO MODE", 
                     command=self.demo_mode,
                     font=('Arial', 10), bg='#f39c12', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="BACKUP FILES", 
                 command=self.backup_files,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="SECURITY LOGS", 
                 command=self.view_logs,
                 font=('Arial', 10), bg='#e67e22', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed
        left_frame = ttk.LabelFrame(content_frame, text="Live Camera - Face Detection & Recognition", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        if self.camera_available:
            self.camera_label = tk.Label(left_frame, text="Camera Ready - Click TEST CAMERA to Start", 
                                       bg='black', fg='white', font=('Arial', 14), justify=tk.CENTER)
        else:
            self.camera_label = tk.Label(left_frame, text="DEMO MODE - Camera Not Available\n\nOpenCV Face Detection Ready", 
                                       bg='black', fg='white', font=('Arial', 14), justify=tk.CENTER)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Recognition results
        results_frame = ttk.LabelFrame(right_frame, text="Face Recognition Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=5, width=45)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Notifications
        notif_frame = ttk.LabelFrame(right_frame, text="Security Notifications", padding=10)
        notif_frame.pack(fill=tk.X, pady=5)
        
        self.notif_text = tk.Text(notif_frame, height=8, width=45)
        self.notif_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(notif_frame, command=self.notif_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.notif_text.config(yscrollcommand=scrollbar.set)
        
        # Logs
        log_frame = ttk.LabelFrame(right_frame, text="Security Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.log_text = tk.Text(log_frame, height=6)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def test_camera(self):
        """Test camera with face detection and recognition"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera not available")
            return
        
        def test_sequence():
            if self.camera_manager.start_camera():
                self.add_notification("🔍 Testing camera with face recognition...")
                
                for i in range(100):  # Test for 100 frames
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Detect faces
                        faces = self.camera_manager.detect_faces(frame)
                        
                        # Process each face
                        recognition_results = []
                        for (x, y, w, h) in faces:
                            face_region = frame[y:y+h, x:x+w]
                            name, color, confidence = self.face_database.recognize_face(face_region)
                            recognition_results.append((x, y, w, h, name, color, confidence))
                            
                            # Draw face box and info
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, f"{name} ({confidence:.2f})", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Update display
                        self.update_camera_display(frame)
                        self.update_recognition_results(recognition_results)
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
                self.add_notification("✅ Camera test completed")
            else:
                self.add_notification("❌ Camera test failed")
        
        threading.Thread(target=test_sequence, daemon=True).start()
    
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
    
    def update_recognition_results(self, results):
        """Update recognition results display"""
        self.results_text.delete(1.0, tk.END)
        
        if results:
            for i, (x, y, w, h, name, color, confidence) in enumerate(results):
                if name == "Unknown":
                    self.results_text.insert(tk.END, f"🚨 Face {i+1}: {name} (Conf: {confidence:.2f})\n", "alert")
                else:
                    self.results_text.insert(tk.END, f"✅ Face {i+1}: {name} (Conf: {confidence:.2f})\n", "known")
        else:
            self.results_text.insert(tk.END, "No faces detected\n")
        
        # Configure text colors
        self.results_text.tag_configure("alert", foreground="red")
        self.results_text.tag_configure("known", foreground="green")
    
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
            self.log_event("Security system ENABLED - OpenCV Face Recognition")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.status_label.config(text="🔴 SECURITY DISABLED", fg='red')
            self.camera_manager.release()
            self.add_notification("Security system deactivated")
            self.log_event("Security system DISABLED")
    
    def start_security_monitoring(self):
        """Start security monitoring"""
        def monitor():
            if self.camera_manager.start_camera():
                last_unknown_alert = 0
                alert_cooldown = 10  # seconds between alerts
                
                while self.security_enabled:
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Detect faces
                        faces = self.camera_manager.detect_faces(frame)
                        
                        unknown_detected = False
                        recognition_results = []
                        
                        for (x, y, w, h) in faces:
                            face_region = frame[y:y+h, x:x+w]
                            name, color, confidence = self.face_database.recognize_face(face_region)
                            recognition_results.append((x, y, w, h, name, color, confidence))
                            
                            # Draw face box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, name, (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            if name == "Unknown":
                                unknown_detected = True
                        
                        # Update displays
                        self.update_camera_display(frame)
                        self.update_recognition_results(recognition_results)
                        
                        # Trigger alerts for unknown faces
                        current_time = time.time()
                        if unknown_detected and (current_time - last_unknown_alert) > alert_cooldown:
                            self.trigger_intruder_alert(frame)
                            last_unknown_alert = current_time
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def trigger_intruder_alert(self, frame):
        """Trigger security alert for unknown face"""
        try:
            os.makedirs('captured_intruders', exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_intruders/intruder_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            self.add_notification("🚨🚨🚨 INTRUDER ALERT! Unknown face detected!")
            self.log_event(f"INTRUDER ALERT: Unknown face captured - {filename}")
            
            # Play sound alert if available
            try:
                import winsound
                for _ in range(3):
                    winsound.Beep(1000, 500)
                    time.sleep(0.2)
            except:
                pass
                
        except Exception as e:
            self.log_event(f"Alert error: {str(e)}")
    
    def register_face(self):
        """Register a new face"""
        if not self.camera_available:
            messagebox.showinfo("No Camera", "Camera required for face registration")
            return
        
        name = simpledialog.askstring("Register Face", "Enter name for this person:")
        if not name:
            return
        
        def capture_and_register():
            if self.camera_manager.start_camera():
                self.add_notification(f"📸 Capturing face for: {name}")
                
                # Let camera stabilize
                time.sleep(1)
                
                ret, frame = self.camera_manager.read_frame()
                if ret:
                    success, message = self.face_database.register_face(frame, name)
                    
                    if success:
                        self.add_notification(f"✅ {message}")
                        self.face_status_label.config(text=f"Known Faces: {len(self.face_database.known_faces)}")
                        messagebox.showinfo("Success", message)
                    else:
                        self.add_notification(f"❌ {message}")
                        messagebox.showerror("Registration Failed", message)
                else:
                    messagebox.showerror("Error", "Could not capture image from camera")
                
                self.camera_manager.release()
            else:
                messagebox.showerror("Error", "Could not start camera")
        
        threading.Thread(target=capture_and_register, daemon=True).start()
    
    def view_faces(self):
        """View registered faces"""
        if not self.face_database.known_faces:
            messagebox.showinfo("No Faces", "No faces registered yet.")
            return
        
        faces_window = tk.Toplevel(self.root)
        faces_window.title("Registered Faces")
        faces_window.geometry("500x600")
        
        tk.Label(faces_window, text="Registered Faces Database", 
                font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(faces_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # List tab
        list_tab = ttk.Frame(notebook)
        notebook.add(list_tab, text="Face List")
        
        listbox = tk.Listbox(list_tab, font=('Arial', 12))
        scrollbar = ttk.Scrollbar(list_tab, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for name in self.face_database.known_faces.keys():
            listbox.insert(tk.END, f"👤 {name}")
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def delete_face():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                name = list(self.face_database.known_faces.keys())[index]
                
                if messagebox.askyesno("Confirm Delete", f"Delete face: {name}?"):
                    del self.face_database.known_faces[name]
                    self.face_database.save_database()
                    listbox.delete(selection)
                    self.face_status_label.config(text=f"Known Faces: {len(self.face_database.known_faces)}")
                    self.add_notification(f"🗑️ Deleted face: {name}")
        
        tk.Button(list_tab, text="Delete Selected", 
                 command=delete_face, bg='#e74c3c', fg='white').pack(pady=10)
    
    def backup_files(self):
        """Backup files"""
        try:
            source_dir = filedialog.askdirectory(title="Select folder to backup")
            if not source_dir:
                return
            
            backup_name = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(source_dir, backup_name)
            
            self.log_event(f"Backup created: {backup_name}")
            self.add_notification(f"✅ Backup completed: {backup_name}")
            messagebox.showinfo("Success", f"Backup created: {backup_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Backup failed: {str(e)}")
    
    def view_logs(self):
        """View security logs in separate window"""
        log_window = tk.Toplevel(self.root)
        log_window.title("Security Logs")
        log_window.geometry("600x400")
        
        log_text = tk.Text(log_window, wrap=tk.WORD)
        log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Copy logs to new window
        log_content = self.log_text.get("1.0", tk.END)
        log_text.insert("1.0", log_content)
        log_text.config(state=tk.DISABLED)
    
    def demo_mode(self):
        """Demo mode without camera"""
        self.add_notification("🔧 Starting demo mode...")
        
        def demo_sequence():
            time.sleep(1)
            self.add_notification("👤 Demo: Face detection active")
            time.sleep(2)
            self.add_notification("✅ Demo: Known face recognized - Admin")
            time.sleep(1)
            self.add_notification("🚨 Demo: Unknown face detected - SECURITY ALERT!")
            time.sleep(1)
            self.add_notification("📸 Demo: Intruder photo captured")
        
        threading.Thread(target=demo_sequence, daemon=True).start()
    
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
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()