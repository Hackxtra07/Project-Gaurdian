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

print("Loading security system with REAL face recognition...")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition library loaded successfully!")
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    print(f"❌ face_recognition not available: {e}")

class RealFaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_face_data()
    
    def load_face_data(self):
        """Load trained face encodings"""
        try:
            if os.path.exists('real_face_data.pkl'):
                with open('real_face_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"✅ Loaded {len(self.known_face_names)} known faces from database")
                return True
        except Exception as e:
            print(f"❌ Error loading face data: {e}")
        return False
    
    def save_face_data(self):
        """Save face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open('real_face_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            print(f"✅ Saved {len(self.known_face_names)} faces to database")
            return True
        except Exception as e:
            print(f"❌ Error saving face data: {e}")
            return False
    
    def register_face(self, image, name):
        """Register a new face using real face recognition"""
        try:
            # Convert BGR to RGB (OpenCV uses BGR, face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find all face locations in the image
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return False, "No face detected. Please ensure your face is clearly visible."
            elif len(face_locations) > 1:
                return False, "Multiple faces detected. Please ensure only one face is visible."
            else:
                # Get face encodings for the detected face
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                if len(face_encodings) == 1:
                    # Add the new face encoding and name
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    self.save_face_data()
                    return True, f"Face registered successfully for '{name}'!"
                else:
                    return False, "Could not extract face features. Please try again."
                    
        except Exception as e:
            return False, f"Error during face registration: {str(e)}"
    
    def recognize_faces(self, image):
        """Recognize faces in image using real face recognition"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        name = self.known_face_names[best_match_index]
                
                face_names.append(name)
            
            return face_locations, face_names
            
        except Exception as e:
            print(f"Face recognition error: {e}")
            return [], []

class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_index = None
        self.find_camera()
    
    def find_camera(self):
        """Find a working camera"""
        print("Searching for cameras...")
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
        self.root.title("Security System - REAL Face Recognition")
        self.root.geometry("1300x850")
        
        self.security_enabled = False
        self.camera_manager = CameraManager()
        
        # Initialize face recognition system
        if FACE_RECOGNITION_AVAILABLE:
            self.face_system = RealFaceRecognition()
        else:
            self.face_system = None
        
        self.camera_available = self.camera_manager.find_camera()
        self.config = self.load_config()
        self.setup_ui()
        
        # Show system status
        if FACE_RECOGNITION_AVAILABLE:
            self.add_notification("✅ REAL Face Recognition System Ready!")
            if self.face_system and self.face_system.known_face_names:
                self.add_notification(f"✅ Loaded {len(self.face_system.known_face_names)} known faces")
        else:
            self.add_notification("❌ Face recognition not available - install face_recognition")
        
        if self.camera_available:
            self.add_notification("✅ Camera detected and ready")
        else:
            self.add_notification("⚠️ Camera not available")
    
    def load_config(self):
        return {
            'capture_path': 'captured_images',
            'alert_on_unknown': True
        }
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_text = "🔒 SECURITY SYSTEM - REAL FACE RECOGNITION" if FACE_RECOGNITION_AVAILABLE else "🔒 SECURITY SYSTEM - FACE RECOGNITION UNAVAILABLE"
        title_label = tk.Label(main_frame, text=title_text, 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔴 SECURITY DISABLED", 
                                   font=('Arial', 12, 'bold'), fg='red', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT)
        
        # Face recognition status
        face_status = "✅ REAL FACE RECOGNITION" if FACE_RECOGNITION_AVAILABLE else "❌ FACE RECOGNITION UNAVAILABLE"
        self.face_status_label = tk.Label(status_frame, text=face_status, 
                                        font=('Arial', 10), fg='green' if FACE_RECOGNITION_AVAILABLE else 'red', bg='#2c3e50')
        self.face_status_label.pack(side=tk.LEFT, padx=20)
        
        # Known faces count
        face_count = len(self.face_system.known_face_names) if self.face_system else 0
        self.faces_count_label = tk.Label(status_frame, text=f"Known Faces: {face_count}", 
                                        font=('Arial', 10), fg='blue', bg='#2c3e50')
        self.faces_count_label.pack(side=tk.LEFT, padx=20)
        
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
        
        if self.camera_available and FACE_RECOGNITION_AVAILABLE:
            tk.Button(control_frame, text="TEST CAMERA & FACES", 
                     command=self.test_camera_faces,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="REGISTER NEW FACE", 
                     command=self.register_new_face,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="MANAGE FACES", 
                     command=self.manage_faces,
                     font=('Arial', 10), bg='#9b59b6', fg='white').pack(side=tk.LEFT, padx=5)
        else:
            tk.Button(control_frame, text="INSTALLATION GUIDE", 
                     command=self.show_installation_guide,
                     font=('Arial', 10), bg='#f39c12', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="BACKUP FILES", 
                 command=self.backup_files,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="VIEW CAPTURES", 
                 command=self.view_captures,
                 font=('Arial', 10), bg='#e67e22', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed
        left_frame = ttk.LabelFrame(content_frame, text="Live Camera with Real Face Recognition", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        if self.camera_available and FACE_RECOGNITION_AVAILABLE:
            self.camera_label = tk.Label(left_frame, text="Camera Ready - Enable Security to Start Real Face Recognition", 
                                       bg='black', fg='white', font=('Arial', 12), justify=tk.CENTER)
            self.camera_label.pack(fill=tk.BOTH, expand=True)
        else:
            help_text = self.get_help_text()
            self.camera_label = tk.Label(left_frame, text=help_text, 
                                       bg='black', fg='white', font=('Arial', 11), justify=tk.LEFT)
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
    
    def get_help_text(self):
        """Get help text based on what's missing"""
        if not FACE_RECOGNITION_AVAILABLE and not self.camera_available:
            return """🔧 SYSTEM SETUP REQUIRED

Missing Components:
❌ Face Recognition Library
❌ Camera Access

To enable full features:

1. Install face_recognition:
   sudo apt install git cmake
   pip install face_recognition

2. Check camera connection
3. Ensure camera permissions

Click 'INSTALLATION GUIDE' for help"""
        
        elif not FACE_RECOGNITION_AVAILABLE:
            return """🔧 FACE RECOGNITION REQUIRED

Camera: ✅ Available
Face Recognition: ❌ Not Installed

To enable face recognition:

Installation Commands:
sudo apt install git cmake
pip install face_recognition

Click 'INSTALLATION GUIDE' for detailed instructions"""
        
        else:
            return """🔧 CAMERA REQUIRED

Face Recognition: ✅ Available  
Camera: ❌ Not Detected

Please:
1. Check camera connection
2. Ensure camera drivers are installed
3. Grant camera permissions"""
    
    def show_installation_guide(self):
        """Show installation guide"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Installation Guide")
        guide_window.geometry("600x500")
        
        guide_text = """🔧 FACE RECOGNITION INSTALLATION GUIDE

Step 1: Install System Dependencies
-----------------------------------
sudo apt update
sudo apt install git cmake build-essential

Step 2: Install Python Packages
------------------------------
pip install face_recognition
pip install opencv-python pillow tkinter numpy

Step 3: Verify Installation
--------------------------
Restart this application after installation.

Troubleshooting:
• If face_recognition fails, try:
  pip install cmake
  pip install dlib
  pip install face_recognition

• For camera issues:
  - Check camera connection
  - Ensure no other app is using camera
  - Try different USB ports

After successful installation:
• Click 'TEST CAMERA & FACES'
• Register faces using 'REGISTER NEW FACE'
• Enable security monitoring"""
        
        text_widget = tk.Text(guide_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert("1.0", guide_text)
        text_widget.config(state=tk.DISABLED)
        
        tk.Button(guide_window, text="Close", 
                 command=guide_window.destroy, bg='#3498db', fg='white').pack(pady=10)
    
    def test_camera_faces(self):
        """Test camera with face recognition"""
        if not FACE_RECOGNITION_AVAILABLE or not self.face_system:
            messagebox.showerror("Not Available", "Face recognition is not available. Please install face_recognition first.")
            return
        
        if not self.camera_available:
            messagebox.showerror("Camera Not Available", "No camera detected.")
            return
        
        def test_sequence():
            if self.camera_manager.start_camera():
                self.add_notification("🔍 Testing camera and face recognition...")
                
                for i in range(100):  # Test for 100 frames
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Perform real face recognition
                        face_locations, face_names = self.face_system.recognize_faces(frame)
                        
                        # Draw results on frame
                        frame = self.draw_recognition_results(frame, face_locations, face_names)
                        self.update_camera_display(frame)
                        
                        # Update results
                        self.update_recognition_display(face_names)
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
                self.add_notification("✅ Camera and face recognition test completed")
            else:
                self.add_notification("❌ Camera test failed")
        
        threading.Thread(target=test_sequence, daemon=True).start()
    
    def draw_recognition_results(self, frame, face_locations, face_names):
        """Draw face recognition results on frame"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates if needed (face_recognition returns relative coordinates)
            # In this case, they should be absolute coordinates
            
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw name
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
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
    
    def update_recognition_display(self, face_names):
        """Update recognition results display"""
        self.results_text.delete(1.0, tk.END)
        
        if face_names:
            for i, name in enumerate(face_names):
                if name == "Unknown":
                    self.results_text.insert(tk.END, f"🚨 Face {i+1}: {name} - INTRUDER!\n", "alert")
                else:
                    self.results_text.insert(tk.END, f"✅ Face {i+1}: {name} - AUTHORIZED\n", "known")
        else:
            self.results_text.insert(tk.END, "No faces detected\n")
        
        # Configure text tags for colors
        self.results_text.tag_configure("alert", foreground="red")
        self.results_text.tag_configure("known", foreground="green")
    
    def toggle_security(self):
        """Toggle security monitoring"""
        if not FACE_RECOGNITION_AVAILABLE or not self.face_system:
            messagebox.showerror("Face Recognition Required", "Please install face_recognition first to enable security monitoring.")
            return
        
        if not self.camera_available:
            messagebox.showerror("Camera Required", "Camera is required for security monitoring.")
            return
        
        self.security_enabled = not self.security_enabled
        
        if self.security_enabled:
            self.security_btn.config(text="DISABLE SECURITY", bg='#2ecc71')
            self.status_label.config(text="🟢 SECURITY ENABLED", fg='green')
            self.start_security_monitoring()
            self.add_notification("🔒 SECURITY MONITORING ACTIVATED - Real Face Recognition Active")
            self.log_event("Security system ENABLED with real face recognition")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.status_label.config(text="🔴 SECURITY DISABLED", fg='red')
            self.stop_security_monitoring()
            self.add_notification("Security system deactivated")
            self.log_event("Security system DISABLED")
    
    def start_security_monitoring(self):
        """Start security monitoring with real face recognition"""
        def monitor():
            if self.camera_manager.start_camera():
                last_unknown_alert = 0
                alert_cooldown = 10  # seconds between unknown face alerts
                
                while self.security_enabled:
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Perform real face recognition
                        face_locations, face_names = self.face_system.recognize_faces(frame)
                        
                        # Update display
                        frame = self.draw_recognition_results(frame, face_locations, face_names)
                        self.update_camera_display(frame)
                        self.update_recognition_display(face_names)
                        
                        # Check for security alerts
                        current_time = time.time()
                        if "Unknown" in face_names and (current_time - last_unknown_alert) > alert_cooldown:
                            self.trigger_intruder_alert(frame)
                            last_unknown_alert = current_time
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def stop_security_monitoring(self):
        """Stop security monitoring"""
        self.camera_manager.release()
    
    def trigger_intruder_alert(self, frame):
        """Trigger alert for unknown face"""
        try:
            os.makedirs(self.config['capture_path'], exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['capture_path']}/INTRUDER_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            
            self.add_notification("🚨🚨🚨 INTRUDER ALERT! Unknown face detected!")
            self.log_event(f"INTRUDER ALERT: Unknown face captured - {filename}")
            
        except Exception as e:
            self.log_event(f"Error capturing intruder: {str(e)}")
    
    def register_new_face(self):
        """Register a new face"""
        if not FACE_RECOGNITION_AVAILABLE or not self.face_system:
            messagebox.showerror("Not Available", "Face recognition is not available.")
            return
        
        if not self.camera_available:
            messagebox.showerror("Camera Required", "Camera is required for face registration.")
            return
        
        # Get name for the new face
        name = simpledialog.askstring("Register Face", "Enter name for this person:")
        if not name:
            return
        
        def capture_and_register():
            if self.camera_manager.start_camera():
                self.add_notification(f"🔍 Capturing face for: {name}")
                
                # Wait for camera to stabilize
                time.sleep(1)
                
                ret, frame = self.camera_manager.read_frame()
                if ret:
                    success, message = self.face_system.register_face(frame, name)
                    
                    if success:
                        self.add_notification(f"✅ {message}")
                        self.faces_count_label.config(text=f"Known Faces: {len(self.face_system.known_face_names)}")
                        messagebox.showinfo("Success", message)
                    else:
                        messagebox.showerror("Registration Failed", message)
                        self.add_notification(f"❌ Face registration failed: {message}")
                else:
                    messagebox.showerror("Error", "Could not capture image from camera")
                    self.add_notification("❌ Camera capture failed")
                
                self.camera_manager.release()
            else:
                messagebox.showerror("Error", "Could not start camera")
        
        threading.Thread(target=capture_and_register, daemon=True).start()
    
    def manage_faces(self):
        """Manage registered faces"""
        if not FACE_RECOGNITION_AVAILABLE or not self.face_system:
            messagebox.showerror("Not Available", "Face recognition is not available.")
            return
        
        if not self.face_system.known_face_names:
            messagebox.showinfo("No Faces", "No faces registered yet.")
            return
        
        manage_window = tk.Toplevel(self.root)
        manage_window.title("Manage Registered Faces")
        manage_window.geometry("400x500")
        
        tk.Label(manage_window, text="Registered Faces", 
                font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Create listbox with scrollbar
        list_frame = ttk.Frame(manage_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        listbox = tk.Listbox(list_frame, font=('Arial', 12))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for name in self.face_system.known_face_names:
            listbox.insert(tk.END, f"👤 {name}")
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons frame
        button_frame = ttk.Frame(manage_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def delete_face():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                name = self.face_system.known_face_names[index]
                
                if messagebox.askyesno("Confirm Delete", f"Delete face: {name}?"):
                    # Remove from face system
                    del self.face_system.known_face_names[index]
                    del self.face_system.known_face_encodings[index]
                    self.face_system.save_face_data()
                    
                    # Update UI
                    listbox.delete(selection)
                    self.faces_count_label.config(text=f"Known Faces: {len(self.face_system.known_face_names)}")
                    self.add_notification(f"🗑️ Deleted face: {name}")
        
        tk.Button(button_frame, text="Delete Selected", 
                 command=delete_face, bg='#e74c3c', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Close", 
                 command=manage_window.destroy, bg='#3498db', fg='white').pack(side=tk.RIGHT, padx=5)
    
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
    
    def view_captures(self):
        """View captured images"""
        capture_path = self.config['capture_path']
        if not os.path.exists(capture_path):
            messagebox.showinfo("No Captures", "No captured images found yet.")
            return
        
        # Open file manager to view captures
        try:
            if os.name == 'nt':  # Windows
                os.startfile(capture_path)
            else:  # Linux/Mac
                os.system(f'xdg-open "{capture_path}"')
            self.add_notification(f"📁 Opened captures folder: {capture_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open file manager: {e}")
    
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