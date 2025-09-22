import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import random
import json
import os
from datetime import datetime
import numpy as np
import argparse

class PreferenceDataCollector:
    def __init__(self, root, args):
        self.root = root
        self.root.title("Aesthetic Preference Data Collection")
        self.root.geometry(f"{args.window_width}x{args.window_height}")
        
        # Data storage
        self.preference_data = []
        self.current_image = None
        self.current_image_path = None  # Store current image path
        self.current_crops = []
        self.crop_info = []  # Store (x, y, w, h) for each crop
        
        # Settings from arguments
        self.crop_size = (args.crop_width, args.crop_height)
        self.num_crops_per_image = args.num_crops
        self.display_size = (args.display_width, args.display_height)
        self.output_file = args.output_file
        self.input_folder = args.input_folder
        self.auto_save_interval = args.auto_save_interval
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Which crop looks more aesthetic?", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Image frames
        self.left_frame = ttk.LabelFrame(main_frame, text="Crop A", padding="10")
        self.left_frame.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.right_frame = ttk.LabelFrame(main_frame, text="Crop B", padding="10")
        self.right_frame.grid(row=1, column=2, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image labels
        self.left_image_label = ttk.Label(self.left_frame)
        self.left_image_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.right_image_label = ttk.Label(self.right_frame)
        self.right_image_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Buttons frame (center)
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=1, padx=20, pady=20)
        
        # Choice buttons
        ttk.Button(buttons_frame, text="Crop A is more aesthetic", 
                  command=lambda: self.record_preference(0),
                  width=20).grid(row=0, column=0, pady=5)
        
        ttk.Button(buttons_frame, text="Crop B is more aesthetic", 
                  command=lambda: self.record_preference(1),
                  width=20).grid(row=1, column=0, pady=5)
        
        ttk.Button(buttons_frame, text="Equal/Skip", 
                  command=lambda: self.record_preference(-1),
                  width=20).grid(row=2, column=0, pady=5)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        ttk.Button(control_frame, text="Load Image", 
                  command=self.load_image).grid(row=0, column=0, padx=5)
        
        ttk.Button(control_frame, text="Generate New Crops", 
                  command=self.generate_crops).grid(row=0, column=1, padx=5)
        
        ttk.Button(control_frame, text="Save Data", 
                  command=self.save_data).grid(row=0, column=2, padx=5)
        
        ttk.Button(control_frame, text="Load Data", 
                  command=self.load_data).grid(row=0, column=3, padx=5)
        
        # Status and info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.status_label = ttk.Label(info_frame, text="Load an image to start collecting preferences")
        self.status_label.grid(row=0, column=0, padx=5)
        
        self.count_label = ttk.Label(info_frame, text="Preferences collected: 0")
        self.count_label.grid(row=0, column=1, padx=20)
        
    def load_image(self):
        """Load an image file"""
        if self.input_folder and os.path.exists(self.input_folder):
            initialdir = self.input_folder
        else:
            initialdir = None
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=initialdir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                self.current_image_path = file_path  # Store the path
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.generate_crops()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def generate_random_crop(self, image, crop_size):
        """Generate a random crop from the image"""
        img_width, img_height = image.size
        crop_width, crop_height = crop_size
        
        # Ensure crop fits within image
        if img_width < crop_width or img_height < crop_height:
            # Resize image to fit crop size
            scale = max(crop_width / img_width, crop_height / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            img_width, img_height = image.size
        
        # Random crop coordinates
        max_x = img_width - crop_width
        max_y = img_height - crop_height
        
        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))
        
        crop = image.crop((x, y, x + crop_width, y + crop_height))
        return crop, (x, y, crop_width, crop_height)
    
    def generate_crops(self):
        """Generate random crops from the current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.current_crops = []
        self.crop_info = []
        
        # Generate random crops
        for i in range(self.num_crops_per_image):
            crop, crop_coords = self.generate_random_crop(self.current_image, self.crop_size)
            self.current_crops.append(crop)
            self.crop_info.append(crop_coords)
        
        # Display crops
        self.display_crops()
        
        self.status_label.config(text="Compare the crops and select your preference")
    
    def display_crops(self):
        """Display the current crops in the UI"""
        if len(self.current_crops) >= 2:
            # Left crop
            left_crop = self.current_crops[0].copy()
            left_crop.thumbnail(self.display_size, Image.Resampling.LANCZOS)
            left_photo = ImageTk.PhotoImage(left_crop)
            self.left_image_label.config(image=left_photo)
            self.left_image_label.image = left_photo  # Keep reference
            
            # Right crop
            right_crop = self.current_crops[1].copy()
            right_crop.thumbnail(self.display_size, Image.Resampling.LANCZOS)
            right_photo = ImageTk.PhotoImage(right_crop)
            self.right_image_label.config(image=right_photo)
            self.right_image_label.image = right_photo  # Keep reference
            
            # Update crop info display
            left_info = f"Crop A: {self.crop_info[0]}"
            right_info = f"Crop B: {self.crop_info[1]}"
            self.left_frame.config(text=left_info)
            self.right_frame.config(text=right_info)
    
    def record_preference(self, choice):
        """Record the user's preference"""
        if len(self.current_crops) < 2:
            messagebox.showwarning("Warning", "Need to generate crops first")
            return
        
        # Create preference record
        preference_record = {
            'timestamp': datetime.now().isoformat(),
            'image_path': self.current_image_path,  # Add image path
            'crop_a': {
                'coordinates': self.crop_info[0],  # (x, y, w, h)
                'image_size': self.current_image.size  # (width, height)
            },
            'crop_b': {
                'coordinates': self.crop_info[1],  # (x, y, w, h)
                'image_size': self.current_image.size  # (width, height)
            },
            'preference': choice,  # 0: A preferred, 1: B preferred, -1: equal/skip
            'crop_size': self.crop_size
        }
        
        # Only save non-skipped preferences
        if choice != -1:
            self.preference_data.append(preference_record)
            self.count_label.config(text=f"Preferences collected: {len(self.preference_data)}")
            
            # Auto-save if interval is set
            if (self.auto_save_interval > 0 and 
                len(self.preference_data) % self.auto_save_interval == 0 and
                self.output_file):
                self.auto_save()
        
        # Generate new crops automatically
        self.generate_crops()
    
    def auto_save(self):
        """Automatically save data at intervals"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.preference_data, f, indent=2)
            self.status_label.config(text=f"Auto-saved {len(self.preference_data)} preferences")
        except Exception as e:
            print(f"Auto-save failed: {str(e)}")
    
    def save_data(self):
        """Save preference data to JSON file"""
        if not self.preference_data:
            messagebox.showwarning("Warning", "No preference data to save")
            return
        
        # Use default output file if specified
        if self.output_file:
            file_path = self.output_file
        else:
            file_path = filedialog.asksaveasfilename(
                title="Save Preference Data",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.preference_data, f, indent=2)
                messagebox.showinfo("Success", f"Saved {len(self.preference_data)} preferences to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def load_data(self):
        """Load existing preference data"""
        file_path = filedialog.askopenfilename(
            title="Load Preference Data",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                self.preference_data.extend(loaded_data)
                self.count_label.config(text=f"Preferences collected: {len(self.preference_data)}")
                messagebox.showinfo("Success", f"Loaded {len(loaded_data)} preferences")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Aesthetic Preference Data Collection Tool")
    
    # Image and crop settings
    parser.add_argument('--crop-width', type=int, default=224,
                       help='Width of generated crops (default: 224)')
    parser.add_argument('--crop-height', type=int, default=224,
                       help='Height of generated crops (default: 224)')
    parser.add_argument('--num-crops', type=int, default=2,
                       help='Number of crops to generate per comparison (default: 2)')
    
    # Display settings
    parser.add_argument('--window-width', type=int, default=1200,
                       help='Window width (default: 1200)')
    parser.add_argument('--window-height', type=int, default=800,
                       help='Window height (default: 800)')
    parser.add_argument('--display-width', type=int, default=300,
                       help='Display width for crop images (default: 300)')
    parser.add_argument('--display-height', type=int, default=300,
                       help='Display height for crop images (default: 300)')
    
    # File settings
    parser.add_argument('--output-file', type=str, default=None,
                       help='Default output file for saving preferences (default: prompt user)')
    parser.add_argument('--input-folder', type=str, default=None,
                       help='Default folder for loading images (default: current directory)')
    
    # Auto-save settings
    parser.add_argument('--auto-save-interval', type=int, default=0,
                       help='Auto-save interval in number of preferences (0 = disabled, default: 0)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    root = tk.Tk()
    app = PreferenceDataCollector(root, args)
    root.mainloop()

if __name__ == "__main__":
    main()