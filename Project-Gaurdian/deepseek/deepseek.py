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

# Platform-specific sound handling
try:
    import winsound
    SOUND_AVAILABLE = True
    SOUND_TYPE = 'winsound'
except ImportError:
    try:
        import os
        SOUND_AVAILABLE = True
        SOUND_TYPE = 'unix'
    except:
        SOUND_AVAILABLE = False
        SOUND_TYPE = 'none'

# Use OpenCV's face detection instead of face_recognition
FACE_RECOGNITION_AVAILABLE = True  # Always available with OpenCV

class SecuritySoftware:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Security System v2.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Security state
        self.security_enabled = False
        self.admin_detected = False
        self.motion_detected = False
        
        # Configuration
        self.config = self.load_config()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = []  # Store face regions for simple recognition
        self.face_threshold = 0.6  # Similarity threshold
        
        # Camera
        self.camera = None
        self.camera_active = False
        
        # Initialize components
        self.setup_ui()
        self.load_face_data()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🔒 ADVANCED SECURITY SYSTEM", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔴 SECURITY DISABLED", 
                                   font=('Arial', 12, 'bold'), fg='red', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT)
        
        self.admin_status_label = tk.Label(status_frame, text="Admin: Not Detected", 
                                         font=('Arial', 10), fg='yellow', bg='#2c3e50')
        self.admin_status_label.pack(side=tk.RIGHT)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.security_btn = tk.Button(control_frame, text="ENABLE SECURITY", 
                                    command=self.toggle_security,
                                    font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                    width=15, height=2)
        self.security_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="REGISTER ADMIN FACE", 
                 command=self.register_face,
                 font=('Arial', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="BACKUP FILES", 
                 command=self.backup_files,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="SETTINGS", 
                 command=self.open_settings,
                 font=('Arial', 10), bg='#f39c12', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel - Camera feed
        left_frame = ttk.LabelFrame(content_frame, text="Live Camera Feed", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera_label = tk.Label(left_frame, text="Camera Inactive", 
                                   bg='black', fg='white', font=('Arial', 14))
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Features
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Face recognition frame
        face_frame = ttk.LabelFrame(right_frame, text="Face Detection", padding=10)
        face_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.face_status_label = tk.Label(face_frame, text="No faces registered", 
                                        font=('Arial', 10), fg='orange')
        self.face_status_label.pack()
        
        tk.Button(face_frame, text="View Registered Faces", 
                 command=self.view_faces).pack(pady=5)
        
        # Motion detection frame
        motion_frame = ttk.LabelFrame(right_frame, text="Motion Detection", padding=10)
        motion_frame.pack(fill=tk.X, pady=5)
        
        self.motion_status_label = tk.Label(motion_frame, text="Motion: Inactive", 
                                          font=('Arial', 10), fg='gray')
        self.motion_status_label.pack()
        
        self.motion_sensitivity = tk.Scale(motion_frame, from_=1, to=100, 
                                         orient=tk.HORIZONTAL, label="Sensitivity")
        self.motion_sensitivity.set(50)
        self.motion_sensitivity.pack(fill=tk.X)
        
        # Notifications frame
        notif_frame = ttk.LabelFrame(right_frame, text="Notifications & Alerts", padding=10)
        notif_frame.pack(fill=tk.X, pady=5)
        
        self.notif_text = tk.Text(notif_frame, height=8, width=40)
        self.notif_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(notif_frame, command=self.notif_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.notif_text.config(yscrollcommand=scrollbar.set)
        
        # Logs frame
        log_frame = ttk.LabelFrame(right_frame, text="Security Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.log_text = tk.Text(log_frame, height=6)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
    def load_config(self):
        default_config = {
            'email_alerts': False,
            'email': '',
            'password': '',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'backup_path': '',
            'motion_threshold': 1000,
            'capture_path': 'captured_images'
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
                    
                self.face_status_label.config(
                    text=f"Registered: {len(self.known_faces)} face(s)",
                    fg='green'
                )
        except Exception as e:
            self.log_event(f"Error loading face data: {str(e)}")
    
    def save_face_data(self):
        with open('face_data.pkl', 'wb') as f:
            pickle.dump(self.known_faces, f)
    
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
                    messagebox.showerror("Camera Error", "Could not access camera")
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
        
        # Initialize frames
        ret, frame1 = self.camera.read()
        if not ret:
            self.log_event("Failed to read from camera")
            return
            
        ret, frame2 = self.camera.read()
        if not ret:
            self.log_event("Failed to read from camera")
            return
        
        while self.camera_active and self.camera.isOpened():
            try:
                # Face detection
                if self.security_enabled and not self.admin_detected:
                    is_admin = self.detect_admin_face(frame1)
                    
                    if is_admin and not self.admin_detected:
                        self.admin_detected = True
                        self.admin_status_label.config(text="Admin: Detected ✅", fg='green')
                        self.add_notification("Admin detected - Security suspended")
                        self.log_event("Admin face recognized - Security temporarily disabled")
                    elif not is_admin and self.admin_detected:
                        self.admin_detected = False
                        self.admin_status_label.config(text="Admin: Not Detected", fg='yellow')
                
                # Motion detection when security enabled and admin not present
                if self.security_enabled and not self.admin_detected:
                    motion_detected = motion_detector.detect_motion(frame1, frame2, 
                                                                  self.motion_sensitivity.get())
                    
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
                    self.log_event("Camera frame read failed")
                    break
                    
                time.sleep(0.03)
                
            except Exception as e:
                self.log_event(f"Camera error: {str(e)}")
                time.sleep(1)
    
    def detect_faces(self, frame):
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def detect_admin_face(self, frame):
        """Simple face recognition by comparing with stored faces"""
        if not self.known_faces:
            return False
            
        current_faces = self.detect_faces(frame)
        if len(current_faces) == 0:
            return False
            
        # For simplicity, assume first detected face
        x, y, w, h = current_faces[0]
        current_face_region = (x, y, w, h)
        
        # Compare with known faces (simple size and position based)
        for known_face in self.known_faces:
            # Simple similarity check based on face size and position
            kx, ky, kw, kh = known_face
            size_similarity = min(w/kw, kw/w)  # Ratio between 0 and 1
            position_similarity = 1.0 - (abs(x-kx) + abs(y-ky)) / (frame.shape[1] + frame.shape[0])
            
            if size_similarity > 0.7 and position_similarity > 0.6:
                return True
                
        return False
    
    def update_camera_feed(self, frame):
        try:
            # Draw face detection results
            if self.security_enabled:
                faces = self.detect_faces(frame)
                for (x, y, w, h) in faces:
                    is_admin = False
                    
                    # Check if this face matches admin
                    for known_face in self.known_faces:
                        kx, ky, kw, kh = known_face
                        size_similarity = min(w/kw, kw/w)
                        position_similarity = 1.0 - (abs(x-kx) + abs(y-ky)) / (frame.shape[1] + frame.shape[0])
                        
                        if size_similarity > 0.7 and position_similarity > 0.6:
                            is_admin = True
                            break
                    
                    color = (0, 255, 0) if is_admin else (0, 0, 255)
                    label = "Admin" if is_admin else "Unknown"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
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
            self.add_notification(f"🚨 INTRUDER DETECTED! Photo saved: {filename}")
            
            # Play alert sound
            threading.Thread(target=self.play_alert_sound, daemon=True).start()
            
            # Send email alert if configured
            if self.config.get('email_alerts', False):
                threading.Thread(target=self.send_email_alert, args=(filename,), daemon=True).start()
                
        except Exception as e:
            self.log_event(f"Error capturing photo: {str(e)}")
    
    def play_alert_sound(self):
        """Cross-platform sound alert"""
        try:
            if not SOUND_AVAILABLE:
                return
                
            if SOUND_TYPE == 'winsound':
                # Windows beep
                for _ in range(3):
                    winsound.Beep(1000, 500)
                    time.sleep(0.2)
            elif SOUND_TYPE == 'unix':
                # Linux/Mac sound (using system beep)
                for _ in range(3):
                    os.system('echo -e "\a"')
                    time.sleep(0.2)
            else:
                # Fallback - print to console
                print("🔔 ALERT! ALERT! ALERT!")
                
        except Exception as e:
            print(f"Sound alert failed: {e}")
    
    def trigger_alerts(self):
        self.add_notification("🚨 MOTION DETECTED! Security alert triggered!")
        self.log_event("MOTION DETECTED - Security alert")
    
    def register_face(self):
        def capture_face():
            try:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                
                if ret:
                    faces = self.detect_faces(frame)
                    
                    if len(faces) == 1:
                        x, y, w, h = faces[0]
                        self.known_faces.append((x, y, w, h))
                        self.save_face_data()
                        
                        self.face_status_label.config(
                            text=f"Registered: {len(self.known_faces)} face(s)",
                            fg='green'
                        )
                        
                        messagebox.showinfo("Success", "Admin face registered successfully!")
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
        register_window.title("Register Admin Face")
        register_window.geometry("300x200")
        
        tk.Label(register_window, text="Face Registration", font=('Arial', 14)).pack(pady=20)
        tk.Label(register_window, text="Look directly at the camera\nMake sure face is well-lit", 
                justify=tk.CENTER).pack(pady=10)
        
        tk.Button(register_window, text="CAPTURE FACE", 
                 command=capture_face, bg='#3498db', fg='white').pack(pady=20)
    
    def backup_files(self):
        try:
            if not self.config.get('backup_path'):
                messagebox.showwarning("Warning", "Please set backup path in settings first")
                return
            
            source_dir = filedialog.askdirectory(title="Select folder to backup")
            if not source_dir:
                return
            
            backup_dir = os.path.join(self.config['backup_path'], 
                                    f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            shutil.copytree(source_dir, backup_dir)
            
            self.log_event(f"Backup created: {backup_dir}")
            self.add_notification(f"✅ Backup completed: {os.path.basename(backup_dir)}")
            messagebox.showinfo("Success", f"Backup created successfully!\n{backup_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Backup failed: {str(e)}")
            self.log_event(f"Backup error: {str(e)}")
    
    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x500")
        
        # Email settings
        email_frame = ttk.LabelFrame(settings_window, text="Email Alerts", padding=10)
        email_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.email_var = tk.BooleanVar(value=self.config.get('email_alerts', False))
        tk.Checkbutton(email_frame, text="Enable Email Alerts", 
                      variable=self.email_var).pack(anchor=tk.W)
        
        tk.Label(email_frame, text="Email:").pack(anchor=tk.W)
        self.email_entry = tk.Entry(email_frame, width=30)
        self.email_entry.insert(0, self.config.get('email', ''))
        self.email_entry.pack(fill=tk.X, pady=2)
        
        tk.Label(email_frame, text="Password:").pack(anchor=tk.W)
        self.password_entry = tk.Entry(email_frame, width=30, show='*')
        self.password_entry.insert(0, self.config.get('password', ''))
        self.password_entry.pack(fill=tk.X, pady=2)
        
        # Backup settings
        backup_frame = ttk.LabelFrame(settings_window, text="Backup Settings", padding=10)
        backup_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(backup_frame, text="Backup Path:").pack(anchor=tk.W)
        backup_path_frame = ttk.Frame(backup_frame)
        backup_path_frame.pack(fill=tk.X, pady=2)
        
        self.backup_entry = tk.Entry(backup_path_frame, width=25)
        self.backup_entry.insert(0, self.config.get('backup_path', ''))
        self.backup_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(backup_path_frame, text="Browse", 
                 command=lambda: self.backup_entry.insert(0, filedialog.askdirectory())).pack(side=tk.RIGHT)
        
        # Save button
        tk.Button(settings_window, text="SAVE SETTINGS", 
                 command=lambda: self.save_settings(settings_window),
                 bg='#27ae60', fg='white').pack(pady=20)
    
    def save_settings(self, window):
        self.config['email_alerts'] = self.email_var.get()
        self.config['email'] = self.email_entry.get()
        self.config['password'] = self.password_entry.get()
        self.config['backup_path'] = self.backup_entry.get()
        
        self.save_config()
        messagebox.showinfo("Success", "Settings saved successfully!")
        window.destroy()
    
    def send_email_alert(self, image_path):
        try:
            if not all([self.config.get('email'), self.config.get('password')]):
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.config['email']
            msg['To'] = self.config['email']
            msg['Subject'] = "🚨 SECURITY ALERT - Intruder Detected"
            
            body = f"""
            Security Alert!
            
            Intruder detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            A photo has been captured and saved to: {image_path}
            
            Please check your security system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                msg.attach(img)
            
            server = smtplib.SMTP(self.config.get('smtp_server', 'smtp.gmail.com'), 
                                self.config.get('smtp_port', 587))
            server.starttls()
            server.login(self.config['email'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            self.log_event("Email alert sent successfully")
            
        except Exception as e:
            self.log_event(f"Email alert failed: {str(e)}")
    
    def view_faces(self):
        view_window = tk.Toplevel(self.root)
        view_window.title("Registered Faces")
        view_window.geometry("300x200")
        
        if not self.known_faces:
            tk.Label(view_window, text="No faces registered", fg='red').pack(pady=50)
        else:
            tk.Label(view_window, text=f"Registered Faces: {len(self.known_faces)}", 
                    font=('Arial', 12, 'bold')).pack(pady=10)
            
            for i, face in enumerate(self.known_faces):
                tk.Label(view_window, text=f"👤 Face {i+1}").pack(pady=2)
    
    def add_notification(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.notif_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.notif_text.see(tk.END)
    
    def log_event(self, event):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {event}\n")
        self.log_text.see(tk.END)
        
        # Also write to file
        with open("security_log.txt", "a") as f:
            f.write(f"[{timestamp}] {event}\n")

class MotionDetector:
    def __init__(self):
        self.previous_frame = None
        
    def detect_motion(self, frame1, frame2, sensitivity):
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
            
            # Compute difference
            frame_diff = cv2.absdiff(gray1, gray2)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Calculate motion level
            motion_level = cv2.countNonZero(thresh)
            
            # Adjust threshold based on sensitivity (1-100 to 500-5000)
            threshold = 500 + (sensitivity * 45)
            
            return motion_level > threshold
            
        except Exception as e:
            return False

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
