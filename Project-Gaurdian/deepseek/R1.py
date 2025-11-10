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
import getpass

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
            
        self.root.title(f"Advanced Security System v3.0 - User: {self.login_system.current_user}")
        self.root.geometry("1400x900")
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
        
    def show_login(self):
        login_window = tk.Toplevel(self.root)
        login_window.title("Security System Login")
        login_window.geometry("300x250")
        login_window.transient(self.root)
        login_window.grab_set()
        login_window.configure(bg='#34495e')
        
        tk.Label(login_window, text="🔒 SECURITY LOGIN", 
                font=('Arial', 16, 'bold'), fg='white', bg='#34495e').pack(pady=20)
        
        # Username
        tk.Label(login_window, text="Username:", fg='white', bg='#34495e').pack()
        username_entry = tk.Entry(login_window, width=20)
        username_entry.pack(pady=5)
        username_entry.focus()
        
        # Password
        tk.Label(login_window, text="Password:", fg='white', bg='#34495e').pack()
        password_entry = tk.Entry(login_window, width=20, show='*')
        password_entry.pack(pady=5)
        
        result = tk.BooleanVar(value=False)
        
        def try_login():
            if self.login_system.authenticate(username_entry.get(), password_entry.get()):
                result.set(True)
                login_window.destroy()
            else:
                messagebox.showerror("Login Failed", "Invalid username or password")
        
        # Login button
        tk.Button(login_window, text="LOGIN", command=try_login, 
                 bg='#27ae60', fg='white', width=15).pack(pady=10)
        
        # Enter key binding
        login_window.bind('<Return>', lambda e: try_login())
        
        self.root.wait_window(login_window)
        return result.get()
        
    def setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Security Tab
        self.security_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.security_tab, text="🔒 Security Dashboard")
        
        # File Manager Tab
        self.file_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.file_tab, text="📁 File Manager")
        
        # Settings Tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="⚙️ Settings")
        
        self.setup_security_tab()
        self.setup_file_manager_tab()
        self.setup_settings_tab()
        
    def setup_security_tab(self):
        # Main frame
        main_frame = ttk.Frame(self.security_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🔒 ADVANCED SECURITY SYSTEM v3.0", 
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
        
        tk.Button(control_frame, text="VIEW CAPTURES", 
                 command=self.view_captures,
                 font=('Arial', 10), bg='#9b59b6', fg='white').pack(side=tk.LEFT, padx=5)
        
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
        
        # Detection History
        history_frame = ttk.LabelFrame(right_frame, text="Recent Detections", padding=10)
        history_frame.pack(fill=tk.X, pady=5)
        
        self.history_list = tk.Listbox(history_frame, height=4)
        self.history_list.pack(fill=tk.BOTH, expand=True)
        
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
        
    def setup_file_manager_tab(self):
        main_frame = ttk.Frame(self.file_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(header_frame, text="📁 SECURITY FILE MANAGER", 
                font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        tk.Button(header_frame, text="Refresh", 
                 command=self.refresh_file_manager).pack(side=tk.RIGHT)
        
        # Path navigation
        path_frame = ttk.Frame(main_frame)
        path_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(path_frame, text="Current Path:").pack(side=tk.LEFT)
        self.current_path = tk.Entry(path_frame, width=50)
        self.current_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.current_path.insert(0, os.getcwd())
        
        tk.Button(path_frame, text="Browse", 
                 command=self.browse_directory).pack(side=tk.RIGHT)
        
        # File list with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for file browser
        columns = ('name', 'size', 'modified', 'type')
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        # Define headings
        self.file_tree.heading('name', text='File Name')
        self.file_tree.heading('size', text='Size')
        self.file_tree.heading('modified', text='Modified')
        self.file_tree.heading('type', text='Type')
        
        # Set column widths
        self.file_tree.column('name', width=300)
        self.file_tree.column('size', width=100)
        self.file_tree.column('modified', width=150)
        self.file_tree.column('type', width=100)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double click
        self.file_tree.bind('<Double-1>', self.on_file_double_click)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(action_frame, text="Open in File Manager", 
                 command=self.open_in_file_manager, bg='#3498db', fg='white').pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="View Image", 
                 command=self.view_selected_image, bg='#9b59b6', fg='white').pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="Delete File", 
                 command=self.delete_selected_file, bg='#e74c3c', fg='white').pack(side=tk.LEFT, padx=2)
        
        # Load initial directory
        self.load_directory(os.getcwd())
        
    def setup_settings_tab(self):
        main_frame = ttk.Frame(self.settings_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # User Management (Admin only)
        if self.login_system.is_admin():
            user_frame = ttk.LabelFrame(main_frame, text="User Management", padding=10)
            user_frame.pack(fill=tk.X, pady=5)
            
            tk.Button(user_frame, text="Add New User", 
                     command=self.add_new_user, bg='#27ae60', fg='white').pack(side=tk.LEFT, padx=5)
            tk.Button(user_frame, text="View Users", 
                     command=self.view_users, bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Email settings
        email_frame = ttk.LabelFrame(main_frame, text="Email Alerts", padding=10)
        email_frame.pack(fill=tk.X, pady=5)
        
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
        backup_frame = ttk.LabelFrame(main_frame, text="Backup Settings", padding=10)
        backup_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(backup_frame, text="Backup Path:").pack(anchor=tk.W)
        backup_path_frame = ttk.Frame(backup_frame)
        backup_path_frame.pack(fill=tk.X, pady=2)
        
        self.backup_entry = tk.Entry(backup_path_frame, width=25)
        self.backup_entry.insert(0, self.config.get('backup_path', ''))
        self.backup_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(backup_path_frame, text="Browse", 
                 command=lambda: self.backup_entry.insert(0, filedialog.askdirectory())).pack(side=tk.RIGHT)
        
        # Security settings
        security_frame = ttk.LabelFrame(main_frame, text="Security Settings", padding=10)
        security_frame.pack(fill=tk.X, pady=5)
        
        self.auto_start_var = tk.BooleanVar(value=self.config.get('auto_start', False))
        tk.Checkbutton(security_frame, text="Auto-start security on login", 
                      variable=self.auto_start_var).pack(anchor=tk.W)
        
        self.sound_alerts_var = tk.BooleanVar(value=self.config.get('sound_alerts', True))
        tk.Checkbutton(security_frame, text="Enable sound alerts", 
                      variable=self.sound_alerts_var).pack(anchor=tk.W)
        
        # Save button
        tk.Button(main_frame, text="SAVE ALL SETTINGS", 
                 command=self.save_all_settings,
                 bg='#27ae60', fg='white', font=('Arial', 12, 'bold')).pack(pady=20)
    
    def load_config(self):
        default_config = {
            'email_alerts': False,
            'email': '',
            'password': '',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'backup_path': '',
            'motion_threshold': 1000,
            'capture_path': 'captured_images',
            'auto_start': False,
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
                    
                self.face_status_label.config(
                    text=f"Registered: {len(self.known_faces)} face(s)",
                    fg='green'
                )
        except Exception as e:
            self.log_event(f"Error loading face data: {str(e)}")
    
    def save_face_data(self):
        with open('face_data.pkl', 'wb') as f:
            pickle.dump(self.known_faces, f)
    
    # File Manager Methods
    def load_directory(self, path):
        self.file_tree.delete(*self.file_tree.get_children())
        self.current_path.delete(0, tk.END)
        self.current_path.insert(0, path)
        
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                stat = os.stat(item_path)
                
                # Get file size
                if os.path.isfile(item_path):
                    size = self.get_file_size(stat.st_size)
                    file_type = os.path.splitext(item)[1] or 'File'
                else:
                    size = 'Folder'
                    file_type = 'Folder'
                
                # Get modification time
                modified = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                self.file_tree.insert('', 'end', values=(item, size, modified, file_type))
        except PermissionError:
            messagebox.showerror("Permission Denied", f"Cannot access: {path}")
    
    def get_file_size(self, size_bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.load_directory(directory)
    
    def on_file_double_click(self, event):
        selected = self.file_tree.selection()
        if selected:
            item = self.file_tree.item(selected[0])
            filename = item['values'][0]
            current_path = self.current_path.get()
            full_path = os.path.join(current_path, filename)
            
            if os.path.isdir(full_path):
                self.load_directory(full_path)
            else:
                self.open_file(full_path)
    
    def open_file(self, filepath):
        try:
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.view_image(filepath)
            else:
                # Try to open with default application
                if os.name == 'nt':  # Windows
                    os.startfile(filepath)
                else:  # Linux/Mac
                    subprocess.run(['xdg-open', filepath])
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open file: {e}")
    
    def open_in_file_manager(self):
        current_path = self.current_path.get()
        try:
            if os.name == 'nt':  # Windows
                os.startfile(current_path)
            else:  # Linux
                subprocess.run(['nautilus', current_path])  # Ubuntu file manager
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open file manager: {e}")
    
    def view_selected_image(self):
        selected = self.file_tree.selection()
        if selected:
            item = self.file_tree.item(selected[0])
            filename = item['values'][0]
            current_path = self.current_path.get()
            full_path = os.path.join(current_path, filename)
            
            if full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.view_image(full_path)
            else:
                messagebox.showwarning("Warning", "Selected file is not an image")
    
    def view_image(self, image_path):
        image_window = tk.Toplevel(self.root)
        image_window.title(f"Image Viewer - {os.path.basename(image_path)}")
        image_window.geometry("800x600")
        
        try:
            # Get file info
            stat = os.stat(image_path)
            file_size = self.get_file_size(stat.st_size)
            modified = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            created = datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Info frame
            info_frame = ttk.Frame(image_window)
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            info_text = f"File: {os.path.basename(image_path)}\nSize: {file_size}\nModified: {modified}\nCreated: {created}"
            tk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)
            
            # Image frame
            img_frame = ttk.Frame(image_window)
            img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Load and display image
            img = Image.open(image_path)
            img.thumbnail((700, 500))
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(img_frame, image=photo)
            img_label.image = photo
            img_label.pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {e}")
            image_window.destroy()
    
    def delete_selected_file(self):
        selected = self.file_tree.selection()
        if selected:
            item = self.file_tree.item(selected[0])
            filename = item['values'][0]
            current_path = self.current_path.get()
            full_path = os.path.join(current_path, filename)
            
            if messagebox.askyesno("Confirm Delete", f"Delete {filename}?"):
                try:
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                    else:
                        shutil.rmtree(full_path)
                    self.load_directory(current_path)
                    self.add_notification(f"Deleted: {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Cannot delete: {e}")
    
    def refresh_file_manager(self):
        self.load_directory(self.current_path.get())
    
    def view_captures(self):
        capture_path = self.config.get('capture_path', 'captured_images')
        if os.path.exists(capture_path):
            self.notebook.select(self.file_tab)
            self.load_directory(capture_path)
        else:
            messagebox.showinfo("Info", "No captures found yet")
    
    # User Management Methods
    def add_new_user(self):
        if not self.login_system.is_admin():
            messagebox.showerror("Permission Denied", "Only admins can create users")
            return
            
        user_window = tk.Toplevel(self.root)
        user_window.title("Add New User")
        user_window.geometry("300x200")
        
        tk.Label(user_window, text="Username:").pack(pady=5)
        username_entry = tk.Entry(user_window, width=20)
        username_entry.pack(pady=5)
        
        tk.Label(user_window, text="Password:").pack(pady=5)
        password_entry = tk.Entry(user_window, width=20, show='*')
        password_entry.pack(pady=5)
        
        tk.Label(user_window, text="Role:").pack(pady=5)
        role_var = tk.StringVar(value='user')
        role_frame = ttk.Frame(user_window)
        role_frame.pack(pady=5)
        tk.Radiobutton(role_frame, text="User", variable=role_var, value='user').pack(side=tk.LEFT)
        tk.Radiobutton(role_frame, text="Admin", variable=role_var, value='admin').pack(side=tk.LEFT)
        
        def create_user():
            if self.login_system.create_user(username_entry.get(), password_entry.get(), role_var.get()):
                messagebox.showinfo("Success", "User created successfully")
                user_window.destroy()
            else:
                messagebox.showerror("Error", "Username already exists")
        
        tk.Button(user_window, text="Create User", command=create_user, 
                 bg='#27ae60', fg='white').pack(pady=10)
    
    def view_users(self):
        if not self.login_system.is_admin():
            messagebox.showerror("Permission Denied", "Only admins can view users")
            return
            
        users_window = tk.Toplevel(self.root)
        users_window.title("System Users")
        users_window.geometry("400x300")
        
        # Create treeview for users
        columns = ('username', 'role', 'created')
        users_tree = ttk.Treeview(users_window, columns=columns, show='headings')
        
        users_tree.heading('username', text='Username')
        users_tree.heading('role', text='Role')
        users_tree.heading('created', text='Created')
        
        users_tree.column('username', width=150)
        users_tree.column('role', width=100)
        users_tree.column('created', width=150)
        
        # Add users to treeview
        for username, info in self.login_system.users.items():
            created = datetime.datetime.fromisoformat(info['created']).strftime('%Y-%m-%d %H:%M')
            users_tree.insert('', 'end', values=(username, info['role'], created))
        
        users_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def save_all_settings(self):
        self.config['email_alerts'] = self.email_var.get()
        self.config['email'] = self.email_entry.get()
        self.config['password'] = self.password_entry.get()
        self.config['backup_path'] = self.backup_entry.get()
        self.config['auto_start'] = self.auto_start_var.get()
        self.config['sound_alerts'] = self.sound_alerts_var.get()
        
        self.save_config()
        messagebox.showinfo("Success", "All settings saved successfully!")
    
    # Existing security methods (truncated for brevity, include all previous security methods)
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
            
            # Add to detection history
            detection_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.history_list.insert(0, f"{detection_time} - Motion Detected")
            if self.history_list.size() > 10:  # Keep only last 10
                self.history_list.delete(10)
            
            self.log_event(f"INTRUDER ALERT: Photo captured - {filename}")
            self.add_notification(f"🚨 INTRUDER DETECTED! Photo saved: {filename}")
            
            # Play alert sound
            if self.config.get('sound_alerts', True):
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
        if app.login_system.current_user:  # Only run if login successful
            root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()
