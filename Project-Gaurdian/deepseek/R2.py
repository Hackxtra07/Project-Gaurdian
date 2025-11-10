import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pickle
import numpy as np
from PIL import Image, ImageTk
import datetime
import shutil
from pathlib import Path
import subprocess
import hashlib

print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Platform-specific sound handling
try:
    import winsound
    SOUND_AVAILABLE = True
    SOUND_TYPE = 'winsound'
except ImportError:
    SOUND_AVAILABLE = False
    SOUND_TYPE = 'none'

class LoginSystem:
    def __init__(self):
        self.users_file = 'users.json'
        self.current_user = None
        self.load_users()
    
    def load_users(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            # Default admin user
            self.users = {
                'admin': {
                    'password': self.hash_password('admin123'),
                    'role': 'admin',
                    'created': datetime.datetime.now().isoformat()
                }
            }
            self.save_users()
    
    def save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=4)
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username, password):
        if username in self.users:
            if self.users[username]['password'] == self.hash_password(password):
                self.current_user = username
                return True
        return False
    
    def create_user(self, username, password, role='user'):
        if username not in self.users:
            self.users[username] = {
                'password': self.hash_password(password),
                'role': role,
                'created': datetime.datetime.now().isoformat()
            }
            self.save_users()
            return True
        return False
    
    def is_admin(self):
        return self.current_user and self.users[self.current_user]['role'] == 'admin'

class SecuritySoftware:
    def __init__(self, root):
        self.root = root
        self.login_system = LoginSystem()
        
        if not self.show_login():
            root.destroy()
            return
            
        self.root.title(f"Security System - User: {self.login_system.current_user}")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Security state
        self.security_enabled = False
        self.admin_detected = False
        self.motion_detected = False
        
        # Configuration
        self.config = self.load_config()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = []
        
        # Camera
        self.camera = None
        self.camera_active = False
        
        # Initialize components
        self.setup_ui()
        self.load_face_data()
        
    def load_config(self):
        default_config = {
            'email_alerts': False,
            'email': '',
            'password': '',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'backup_path': '',
            'capture_path': 'captured_images',
            'sound_alerts': True
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
    
    def show_login(self):
        login_window = tk.Toplevel(self.root)
        login_window.title("Security System Login")
        login_window.geometry("300x200")
        login_window.transient(self.root)
        login_window.grab_set()
        login_window.configure(bg='#34495e')
        
        tk.Label(login_window, text="🔒 SECURITY LOGIN", 
                font=('Arial', 16, 'bold'), fg='white', bg='#34495e').pack(pady=20)
        
        tk.Label(login_window, text="Username:", fg='white', bg='#34495e').pack()
        username_entry = tk.Entry(login_window, width=20)
        username_entry.pack(pady=5)
        username_entry.insert(0, 'admin')  # Default username
        username_entry.focus()
        
        tk.Label(login_window, text="Password:", fg='white', bg='#34495e').pack()
        password_entry = tk.Entry(login_window, width=20, show='*')
        password_entry.pack(pady=5)
        password_entry.insert(0, 'admin123')  # Default password
        
        result = tk.BooleanVar(value=False)
        
        def try_login():
            if self.login_system.authenticate(username_entry.get(), password_entry.get()):
                result.set(True)
                login_window.destroy()
            else:
                messagebox.showerror("Login Failed", "Invalid username or password")
        
        tk.Button(login_window, text="LOGIN", command=try_login, 
                 bg='#27ae60', fg='white', width=15).pack(pady=10)
        
        login_window.bind('<Return>', lambda e: try_login())
        
        self.root.wait_window(login_window)
        return result.get()

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
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.security_btn = tk.Button(control_frame, text="ENABLE SECURITY", 
                                    command=self.toggle_security,
                                    font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                    width=15, height=2)
        self.security_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="REGISTER FACE", 
                 command=self.register_face,
                 font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="BACKUP FILES", 
                 command=self.backup_files,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera feed
        left_frame = ttk.LabelFrame(content_frame, text="Live Camera", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera_label = tk.Label(left_frame, text="Camera Inactive", 
                                   bg='black', fg='white', font=('Arial', 14))
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Face recognition
        face_frame = ttk.LabelFrame(right_frame, text="Face Detection", padding=10)
        face_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.face_status_label = tk.Label(face_frame, text="No faces registered", 
                                        font=('Arial', 10), fg='orange')
        self.face_status_label.pack()
        
        # Motion detection
        motion_frame = ttk.LabelFrame(right_frame, text="Motion Detection", padding=10)
        motion_frame.pack(fill=tk.X, pady=5)
        
        self.motion_status_label = tk.Label(motion_frame, text="Motion: Inactive", 
                                          font=('Arial', 10), fg='gray')
        self.motion_status_label.pack()
        
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

    def toggle_security(self):
        self.security_enabled = not self.security_enabled
        
        if self.security_enabled:
            self.security_btn.config(text="DISABLE SECURITY", bg='#2ecc71')
            self.status_label.config(text="🟢 SECURITY ENABLED", fg='green')
            self.start_camera()
            self.log_event("Security system ENABLED")
            self.add_notification("Security system activated")
        else:
            self.security_btn.config(text="ENABLE SECURITY", bg='#e74c3c')
            self.status_label.config(text="🔴 SECURITY DISABLED", fg='red')
            self.stop_camera()
            self.log_event("Security system DISABLED")
            self.add_notification("Security system deactivated")
    
    def start_camera(self):
        if not self.camera_active:
            self.camera_active = True
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    # Try different camera indices
                    for i in range(1, 4):
                        self.camera = cv2.VideoCapture(i)
                        if self.camera.isOpened():
                            break
                
                if not self.camera.isOpened():
                    messagebox.showerror("Camera Error", "Could not access any camera")
                    self.camera_active = False
                    return
                
                self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
                self.camera_thread.start()
            except Exception as e:
                messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
                self.camera_active = False
    
    def stop_camera(self):
        self.camera_active = False
        if self.camera:
            self.camera.release()
        self.camera_label.config(text="Camera Inactive", image='')
    
    def camera_loop(self):
        motion_detector = MotionDetector()
        ret, frame1 = self.camera.read()
        ret, frame2 = self.camera.read()
        
        while self.camera_active and self.camera.isOpened():
            try:
                # Motion detection
                if self.security_enabled:
                    motion_detected = motion_detector.detect_motion(frame1, frame2)
                    
                    if motion_detected and not self.motion_detected:
                        self.motion_detected = True
                        self.motion_status_label.config(text="Motion: DETECTED! 🔴", fg='red')
                        self.capture_intruder_photo(frame1)
                        self.trigger_alerts()
                    elif not motion_detected and self.motion_detected:
                        self.motion_detected = False
                        self.motion_status_label.config(text="Motion: Monitoring ✅", fg='green')
                
                # Update camera feed
                self.update_camera_feed(frame1)
                
                # Read next frame
                frame1 = frame2
                ret, frame2 = self.camera.read()
                if not ret:
                    break
                    
                time.sleep(0.03)
                
            except Exception as e:
                self.log_event(f"Camera error: {str(e)}")
                time.sleep(1)
    
    def update_camera_feed(self, frame):
        try:
            # Convert to PhotoImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(pil_image)
            
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo
            
        except Exception as e:
            pass
    
    def capture_intruder_photo(self, frame):
        try:
            os.makedirs(self.config['capture_path'], exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['capture_path']}/intruder_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            self.log_event(f"INTRUDER ALERT: Photo captured - {filename}")
            self.add_notification(f"🚨 INTRUDER DETECTED! Photo saved")
            
        except Exception as e:
            self.log_event(f"Error capturing photo: {str(e)}")
    
    def trigger_alerts(self):
        self.add_notification("🚨 MOTION DETECTED! Security alert!")
        self.log_event("MOTION DETECTED - Security alert")
    
    def register_face(self):
        def capture_face():
            try:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    if len(faces) == 1:
                        x, y, w, h = faces[0]
                        self.known_faces.append((x, y, w, h))
                        self.save_face_data()
                        
                        self.face_status_label.config(
                            text=f"Registered: {len(self.known_faces)} face(s)",
                            fg='green'
                        )
                        messagebox.showinfo("Success", "Face registered successfully!")
                    else:
                        messagebox.showerror("Error", "Please ensure exactly one face is visible")
                else:
                    messagebox.showerror("Error", "Could not access camera")
                    
                cap.release()
                register_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Face registration failed: {str(e)}")
                if 'cap' in locals():
                    cap.release()
                register_window.destroy()
        
        register_window = tk.Toplevel(self.root)
        register_window.title("Register Face")
        register_window.geometry("300x200")
        
        tk.Label(register_window, text="Face Registration", font=('Arial', 14)).pack(pady=20)
        tk.Label(register_window, text="Look directly at the camera", 
                justify=tk.CENTER).pack(pady=10)
        
        tk.Button(register_window, text="CAPTURE FACE", 
                 command=capture_face, bg='#3498db', fg='white').pack(pady=20)
    
    def backup_files(self):
        try:
            source_dir = filedialog.askdirectory(title="Select folder to backup")
            if not source_dir:
                return
            
            backup_dir = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(source_dir, backup_dir)
            
            self.log_event(f"Backup created: {backup_dir}")
            self.add_notification(f"✅ Backup completed")
            messagebox.showinfo("Success", f"Backup created: {backup_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Backup failed: {str(e)}")
    
    def add_notification(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.notif_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.notif_text.see(tk.END)
    
    def log_event(self, event):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {event}\n")
        self.log_text.see(tk.END)

class MotionDetector:
    def detect_motion(self, frame1, frame2):
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
            
            frame_diff = cv2.absdiff(gray1, gray2)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            motion_level = cv2.countNonZero(thresh)
            return motion_level > 1000
            
        except Exception as e:
            return False

def main():
    try:
        root = tk.Tk()
        app = SecuritySoftware(root)
        if app.login_system.current_user:
            root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Failed to start: {e}")

if __name__ == "__main__":
    main()