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

print("Loading security system...")

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
        self.root.title("Security System - No Camera Required")
        self.root.geometry("1000x700")
        
        # Security state
        self.security_enabled = False
        self.camera_manager = CameraManager()
        self.camera_available = self.camera_manager.find_camera()
        
        # Configuration
        self.config = self.load_config()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = []
        
        # Initialize components
        self.setup_ui()
        self.load_face_data()
        
        # Show camera status
        if self.camera_available:
            self.add_notification("✅ Camera detected and ready")
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
    
    def load_face_data(self):
        try:
            if os.path.exists('face_data.pkl'):
                with open('face_data.pkl', 'rb') as f:
                    self.known_faces = pickle.load(f)
                if hasattr(self, 'face_status_label'):
                    self.face_status_label.config(text=f"Registered: {len(self.known_faces)} face(s)", fg='green')
        except Exception as e:
            self.log_event(f"Error loading face data: {str(e)}")
    
    def save_face_data(self):
        with open('face_data.pkl', 'wb') as f:
            pickle.dump(self.known_faces, f)

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🔒 SECURITY SYSTEM", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔴 SECURITY DISABLED", 
                                   font=('Arial', 12, 'bold'), fg='red', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT)
        
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
        
        # Only enable camera features if camera is available
        if self.camera_available:
            tk.Button(control_frame, text="TEST CAMERA", 
                     command=self.test_camera,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
            
            tk.Button(control_frame, text="REGISTER FACE", 
                     command=self.register_face,
                     font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        else:
            # Demo buttons when no camera
            tk.Button(control_frame, text="DEMO MODE", 
                     command=self.demo_mode,
                     font=('Arial', 10), bg='#f39c12', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="BACKUP FILES", 
                 command=self.backup_files,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="VIEW LOGS", 
                 command=self.view_logs,
                 font=('Arial', 10), bg='#9b59b6', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed or demo display
        left_frame = ttk.LabelFrame(content_frame, text="Live Feed" if self.camera_available else "Security Dashboard", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        if self.camera_available:
            self.camera_label = tk.Label(left_frame, text="Camera Ready - Click 'TEST CAMERA'", 
                                       bg='black', fg='white', font=('Arial', 14))
            self.camera_label.pack(fill=tk.BOTH, expand=True)
        else:
            demo_text = """🔒 SECURITY SYSTEM - DEMO MODE

Camera not detected. The system is running in demo mode.

Available Features:
✅ File Backup
✅ Security Logging  
✅ Notifications
✅ User Interface

To enable camera features:
1. Check camera connection
2. Run camera_test.py
3. Ensure proper permissions"""
            
            self.demo_label = tk.Label(left_frame, text=demo_text, 
                                     bg='black', fg='white', font=('Arial', 12), justify=tk.LEFT)
            self.demo_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Face recognition (only if camera available)
        if self.camera_available:
            face_frame = ttk.LabelFrame(right_frame, text="Face Detection", padding=10)
            face_frame.pack(fill=tk.X, pady=(0, 5))
            
            self.face_status_label = tk.Label(face_frame, text="No faces registered", 
                                            font=('Arial', 10), fg='orange')
            self.face_status_label.pack()
        
        # Motion detection
        motion_frame = ttk.LabelFrame(right_frame, text="Security Monitoring", padding=10)
        motion_frame.pack(fill=tk.X, pady=5)
        
        self.motion_status_label = tk.Label(motion_frame, text="Status: Ready", 
                                          font=('Arial', 10), fg='gray')
        self.motion_status_label.pack()
        
        # Notifications
        notif_frame = ttk.LabelFrame(right_frame, text="Notifications", padding=10)
        notif_frame.pack(fill=tk.X, pady=5)
        
        self.notif_text = tk.Text(notif_frame, height=8, width=40)
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
        """Test camera with preview"""
        if not self.camera_available:
            messagebox.showwarning("Camera Not Available", "No camera detected on this system.")
            return
        
        def show_preview():
            if self.camera_manager.start_camera():
                self.add_notification("✅ Camera test started")
                
                for _ in range(50):  # Show 50 frames (about 5 seconds)
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        self.update_camera_display(frame)
                    time.sleep(0.1)
                
                self.camera_manager.release()
                self.add_notification("✅ Camera test completed")
            else:
                self.add_notification("❌ Camera test failed")
        
        threading.Thread(target=show_preview, daemon=True).start()

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
            self.motion_status_label.config(text="Status: ACTIVE", fg='green')
            
            if self.camera_available:
                self.start_security_monitoring()
            
            self.log_event("Security system ENABLED")
            self.add_notification("Security system activated")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.status_label.config(text="🔴 SECURITY DISABLED", fg='red')
            self.motion_status_label.config(text="Status: INACTIVE", fg='red')
            
            if self.camera_available:
                self.stop_security_monitoring()
            
            self.log_event("Security system DISABLED")
            self.add_notification("Security system deactivated")

    def start_security_monitoring(self):
        """Start security monitoring with camera"""
        if not self.camera_available:
            return
        
        def monitor():
            if self.camera_manager.start_camera():
                self.add_notification("🔒 Security monitoring started")
                
                while self.security_enabled:
                    ret, frame = self.camera_manager.read_frame()
                    if ret:
                        self.update_camera_display(frame)
                        
                        # Simulate motion detection (every 3 seconds)
                        if int(time.time()) % 3 == 0:
                            self.simulate_motion_detection()
                    
                    time.sleep(0.1)
                
                self.camera_manager.release()
        
        threading.Thread(target=monitor, daemon=True).start()

    def stop_security_monitoring(self):
        """Stop security monitoring"""
        self.camera_manager.release()

    def simulate_motion_detection(self):
        """Simulate motion detection for demo"""
        if self.security_enabled and np.random.random() > 0.7:  # 30% chance
            self.add_notification("🚨 Motion detected!")
            self.log_event("Motion detected - Security alert")
            
            # Capture "intruder" photo (create a blank image)
            try:
                os.makedirs(self.config['capture_path'], exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.config['capture_path']}/alert_{timestamp}.jpg"
                
                # Create a demo image
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(filename, img)
                
                self.log_event(f"Alert captured: {filename}")
            except Exception as e:
                self.log_event(f"Error capturing alert: {str(e)}")

    def demo_mode(self):
        """Run demo mode without camera"""
        self.add_notification("🔧 Starting demo mode...")
        
        def demo_sequence():
            time.sleep(1)
            self.add_notification("🔒 Demo: Security monitoring")
            time.sleep(2)
            self.add_notification("🚨 Demo: Motion detected!")
            time.sleep(1)
            self.add_notification("✅ Demo: Alert recorded")
        
        threading.Thread(target=demo_sequence, daemon=True).start()

    def register_face(self):
        """Register face (simplified version)"""
        if not self.camera_available:
            messagebox.showinfo("Camera Required", "Face registration requires a camera.")
            return
        
        self.add_notification("Starting face registration...")
        # Simplified face registration
        self.known_faces.append("demo_face")
        self.save_face_data()
        self.face_status_label.config(text="Registered: Demo Face", fg='green')
        messagebox.showinfo("Success", "Face registration completed (demo)")

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
        """View security logs"""
        log_window = tk.Toplevel(self.root)
        log_window.title("Security Logs")
        log_window.geometry("600x400")
        
        log_text = tk.Text(log_window, wrap=tk.WORD)
        log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add all logs to the window
        log_content = self.log_text.get("1.0", tk.END)
        log_text.insert("1.0", log_content)
        log_text.config(state=tk.DISABLED)

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