import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
import json
import pickle
import numpy as np
from PIL import Image, ImageTk
import datetime
import shutil
import face_recognition
import math

print("Loading security system with face recognition...")

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_face_data()
    
    def load_face_data(self):
        """Load trained face encodings"""
        try:
            if os.path.exists('face_encodings.pkl'):
                with open('face_encodings.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"✅ Loaded {len(self.known_face_names)} known faces")
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
            with open('face_encodings.pkl', 'wb') as f:
                pickle.dump(data, f)
            print("✅ Face data saved successfully")
            return True
        except Exception as e:
            print(f"❌ Error saving face data: {e}")
            return False
    
    def register_face(self, image, name):
        """Register a new face"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_encodings) == 1:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                self.save_face_data()
                return True, "Face registered successfully!"
            elif len(face_encodings) > 1:
                return False, "Multiple faces detected. Please ensure only one face is visible."
            else:
                return False, "No face detected. Please try again."
                
        except Exception as e:
            return False, f"Error registering face: {str(e)}"
    
    def recognize_face(self, image):
        """Recognize faces in image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Calculate face distance
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
        """Find a working camera by testing different indices"""
        print("Searching for available cameras...")
        for i in range(5):  # Test indices 0-4
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Found working camera at index {i}")
                        self.camera_index = i
                        cap.release()
                        return True
                    cap.release()
            except Exception as e:
                print(f"Camera index {i} error: {e}")
        
        print("❌ No working camera found")
        return False
    
    def start_camera(self):
        """Start the camera"""
        if self.camera_index is None:
            if not self.find_camera():
                return False
        
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            # Set lower resolution for better compatibility
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if self.camera.isOpened():
                print(f"✅ Camera started successfully at index {self.camera_index}")
                return True
            return False
        except Exception as e:
            print(f"Camera start error: {e}")
            return False
    
    def read_frame(self):
        """Read a frame from camera"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            return ret, frame
        return False, None
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None

class SecuritySoftware:
    def __init__(self, root):
        self.root = root
        self.root.title("Security System with Face Recognition")
        self.root.geometry("1200x800")
        
        # Security state
        self.security_enabled = False
        self.camera_manager = CameraManager()
        self.face_system = FaceRecognitionSystem()
        self.camera_available = self.camera_manager.find_camera()
        
        # Configuration
        self.config = self.load_config()
        
        # Initialize components
        self.setup_ui()
        
        # Show camera status
        if self.camera_available:
            self.add_notification("✅ Camera detected and ready")
            self.add_notification(f"✅ Loaded {len(self.face_system.known_face_names)} known faces")
        else:
            self.add_notification("⚠️ Camera not available - running in demo mode")
    
    def load_config(self):
        default_config = {
            'capture_path': 'captured_images',
        }
        
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    return json.load(f)
        except:
            pass
            
        return default_config
    
    def save_config(self):
        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🔒 SECURITY SYSTEM WITH FACE RECOGNITION", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔴 SECURITY DISABLED", 
                                   font=('Arial', 12, 'bold'), fg='red', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT)
        
        # Face recognition status
        face_count = len(self.face_system.known_face_names)
        self.face_status_label = tk.Label(status_frame, text=f"Known Faces: {face_count}", 
                                        font=('Arial', 10), fg='blue', bg='#2c3e50')
        self.face_status_label.pack(side=tk.LEFT, padx=20)
        
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
        
        # Camera features
        if self.camera_available:
            tk.Button(control_frame, text="TEST CAMERA", 
                     command=self.test_camera,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="REGISTER FACE", 
                     command=self.register_face_dialog,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="VIEW FACES", 
                     command=self.view_registered_faces,
                     font=('Arial', 10), bg='#9b59b6', fg='white').pack(side=tk.LEFT, padx=5)
        else:
            tk.Button(control_frame, text="DEMO MODE", 
                     command=self.demo_mode,
                     font=('Arial', 10), bg='#f39c12', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="BACKUP FILES", 
                 command=self.backup_files,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed
        left_frame = ttk.LabelFrame(content_frame, text="Live Camera Feed with Face Recognition", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        if self.camera_available:
            self.camera_label = tk.Label(left_frame, text="Camera Ready - Enable Security to Start", 
                                       bg='black', fg='white', font=('Arial', 14))
            self.camera_label.pack(fill=tk.BOTH, expand=True)
        else:
            demo_text = """🔒 FACE RECOGNITION SYSTEM

Camera not detected. System features:

✅ Real Face Recognition
✅ Multiple Face Detection  
✅ Face Database Management
✅ Security Monitoring
✅ File Backup

To enable camera features, ensure:
1. Camera is connected
2. Proper drivers installed
3. App has camera permissions"""
            
            self.demo_label = tk.Label(left_frame, text=demo_text, 
                                     bg='black', fg='white', font=('Arial', 12), justify=tk.LEFT)
            self.demo_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Recognition results
        results_frame = ttk.LabelFrame(right_frame, text="Recognition Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=4, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Notifications
        notif_frame = ttk.LabelFrame(right_frame, text="Security Notifications", padding=10)
        notif_frame.pack(fill=tk.X, pady=5)
        
        self.notif_text = tk.Text(notif_frame, height=6, width=40)
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
        """Test camera with face recognition preview"""
        if not self.camera_available:
            messagebox.showwarning("Camera Not Available", "No camera detected.")
            return
        
        def show_preview():
            if self.camera_manager.start_camera():
                self.add_notification("✅ Camera test with face recognition started")
                
                for _ in range(100):  # Show 100 frames
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Perform face recognition
                        face_locations, face_names = self.face_system.recognize_face(frame)
                        
                        # Draw recognition results
                        frame_with_faces = self.draw_face_boxes(frame, face_locations, face_names)
                        self.update_camera_display(frame_with_faces)
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
                self.add_notification("✅ Camera test completed")
            else:
                self.add_notification("❌ Camera test failed")
        
        threading.Thread(target=show_preview, daemon=True).start()

    def draw_face_boxes(self, frame, face_locations, face_names):
        """Draw face recognition boxes and labels"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back face locations (face_recognition returns scaled coordinates)
            top = int(top * 1)  # Adjust if needed
            right = int(right * 1)
            bottom = int(bottom * 1)
            left = int(left * 1)
            
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
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

    def toggle_security(self):
        self.security_enabled = not self.security_enabled
        
        if self.security_enabled:
            self.security_btn.config(text="DISABLE SECURITY", bg='#2ecc71')
            self.status_label.config(text="🟢 SECURITY ENABLED", fg='green')
            
            if self.camera_available:
                self.start_security_monitoring()
            
            self.log_event("Security system ENABLED")
            self.add_notification("🔒 Security monitoring started with face recognition")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.status_label.config(text="🔴 SECURITY DISABLED", fg='red')
            
            if self.camera_available:
                self.stop_security_monitoring()
            
            self.log_event("Security system DISABLED")
            self.add_notification("Security system deactivated")

    def start_security_monitoring(self):
        """Start security monitoring with face recognition"""
        if not self.camera_available:
            return
        
        def monitor():
            if self.camera_manager.start_camera():
                last_alert_time = 0
                
                while self.security_enabled:
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        # Perform face recognition
                        face_locations, face_names = self.face_system.recognize_face(frame)
                        
                        # Update recognition results
                        self.update_recognition_results(face_names)
                        
                        # Draw face boxes
                        frame_with_faces = self.draw_face_boxes(frame, face_locations, face_names)
                        self.update_camera_display(frame_with_faces)
                        
                        # Security alerts for unknown faces
                        current_time = time.time()
                        if "Unknown" in face_names and (current_time - last_alert_time) > 10:
                            self.trigger_intruder_alert(frame)
                            last_alert_time = current_time
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
        
        threading.Thread(target=monitor, daemon=True).start()

    def update_recognition_results(self, face_names):
        """Update recognition results display"""
        self.results_text.delete(1.0, tk.END)
        if face_names:
            for i, name in enumerate(face_names):
                status = "✅ KNOWN" if name != "Unknown" else "🚨 UNKNOWN"
                self.results_text.insert(tk.END, f"Face {i+1}: {name} {status}\n")
        else:
            self.results_text.insert(tk.END, "No faces detected")

    def trigger_intruder_alert(self, frame):
        """Trigger alert for unknown face"""
        try:
            os.makedirs(self.config['capture_path'], exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['capture_path']}/intruder_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            
            self.add_notification("🚨 INTRUDER ALERT! Unknown face detected!")
            self.log_event(f"INTRUDER: Unknown face captured - {filename}")
            
        except Exception as e:
            self.log_event(f"Error capturing intruder: {str(e)}")

    def stop_security_monitoring(self):
        """Stop security monitoring"""
        self.camera_manager.release()

    def register_face_dialog(self):
        """Dialog for registering new face"""
        if not self.camera_available:
            messagebox.showinfo("Camera Required", "Face registration requires a camera.")
            return
        
        # Get name for new face
        name = tk.simpledialog.askstring("Register Face", "Enter name for this face:")
        if not name:
            return
        
        def capture_and_register():
            if self.camera_manager.start_camera():
                # Wait for camera to stabilize
                time.sleep(1)
                
                ret, frame = self.camera_manager.read_frame()
                if ret:
                    success, message = self.face_system.register_face(frame, name)
                    
                    if success:
                        self.add_notification(f"✅ Face registered: {name}")
                        self.face_status_label.config(text=f"Known Faces: {len(self.face_system.known_face_names)}")
                        messagebox.showinfo("Success", message)
                    else:
                        messagebox.showerror("Error", message)
                else:
                    messagebox.showerror("Error", "Could not capture image from camera")
                
                self.camera_manager.release()
        
        threading.Thread(target=capture_and_register, daemon=True).start()

    def view_registered_faces(self):
        """Show registered faces"""
        if not self.face_system.known_face_names:
            messagebox.showinfo("No Faces", "No faces registered yet.")
            return
        
        faces_window = tk.Toplevel(self.root)
        faces_window.title("Registered Faces")
        faces_window.geometry("300x400")
        
        tk.Label(faces_window, text="Registered Faces", 
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        listbox = tk.Listbox(faces_window)
        for name in self.face_system.known_face_names:
            listbox.insert(tk.END, f"👤 {name}")
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Delete button
        def delete_face():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                name = self.face_system.known_face_names[index]
                if messagebox.askyesno("Delete Face", f"Delete face: {name}?"):
                    del self.face_system.known_face_names[index]
                    del self.face_system.known_face_encodings[index]
                    self.face_system.save_face_data()
                    listbox.delete(selection)
                    self.face_status_label.config(text=f"Known Faces: {len(self.face_system.known_face_names)}")
                    self.add_notification(f"🗑️ Deleted face: {name}")
        
        tk.Button(faces_window, text="Delete Selected", 
                 command=delete_face, bg='#e74c3c', fg='white').pack(pady=10)

    def demo_mode(self):
        """Demo mode without camera"""
        self.add_notification("🔧 Starting face recognition demo...")
        
        def demo_sequence():
            time.sleep(1)
            self.add_notification("👤 Demo: Face detection active")
            time.sleep(2)
            self.add_notification("✅ Demo: Known face recognized - Admin")
            time.sleep(1)
            self.add_notification("🚨 Demo: Unknown face detected - ALERT!")
        
        threading.Thread(target=demo_sequence, daemon=True).start()

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