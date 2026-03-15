import tkinter as tk
import random
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

# ===== CONFIGURATION =====
BASE_BOX_WIDTH = 80
BASE_BOX_HEIGHT = 80
BACKGROUND_IMAGE_PATH = "Screenshot from 2026-03-08 15-23-01.png"   # <-- change to your image file
# =========================

class Box:
    def __init__(self, canvas, x, y, image_path, app):
        self.canvas = canvas
        self.app = app
        self.original_image = Image.open(image_path).convert("RGBA")
        self.base_width = BASE_BOX_WIDTH
        self.base_height = BASE_BOX_HEIGHT

        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 2
        self.state = 'moving'
        self.assigned_slot = None
        self.target_x = None
        self.target_y = None
        self.reached_slot = False

        # Compute initial size based on starting y
        scale = 0.5 + (self.y / self.app.height) * 1.0
        self.width = max(10, int(self.base_width * scale))
        self.height = max(10, int(self.base_height * scale))
        resized = self.original_image.resize((self.width, self.height),
                                             Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.id = canvas.create_image(x, y, image=self.photo)

    def update_image(self):
        scale = 0.5 + (self.y / self.app.height) * 1.0
        new_width = max(10, int(self.base_width * scale))
        new_height = max(10, int(self.base_height * scale))
        self.width = new_width
        self.height = new_height
        resized = self.original_image.resize((new_width, new_height),
                                             Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self.id, image=self.photo)

    def update_position(self):
        if self.state == 'stopped':
            return

        # If assigned to a slot and not yet reached, steer towards it
        if self.assigned_slot is not None and not self.reached_slot:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            self.vx = dx * 0.1
            self.vy = dy * 0.1
            if abs(dx) < 5 and abs(dy) < 5:
                self.x = self.target_x
                self.y = self.target_y
                self.reached_slot = True
                self.vx = 0
                self.vy = 2   # continue downward after reaching slot
                self.canvas.coords(self.id, self.x, self.y)
                self.update_image()
        else:
            # Not assigned or already reached slot: move straight down
            self.vx = 0
            self.vy = 2

        self.x += self.vx
        self.y += self.vy

        # Slot assignment (only once, when crossing threshold)
        if self.assigned_slot is None and self.y > self.app.threshold_y:
            self.assign_slot()

        self.update_image()
        self.canvas.coords(self.id, self.x, self.y)

        # Remove box if it falls off screen
        if self.y > self.app.height + 100:
            self.canvas.delete(self.id)
            self.app.boxes.remove(self)

    def assign_slot(self):
        slot_index = random.randint(0, 3)
        self.assigned_slot = slot_index
        self.target_x, self.target_y = self.app.slots[slot_index]

    def contains_point(self, px, py):
        half_w = self.width / 2
        half_h = self.height / 2
        return (self.x - half_w <= px <= self.x + half_w and
                self.y - half_h <= py <= self.y + half_h)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Box Conveyor Simulation with Video Recording")

        # Canvas dimensions
        self.width = 800
        self.height = 600
        self.threshold_y = self.height * 0.4

        self.slots = [
            (self.width * 0.30, self.height * 0.4),
            (self.width * 0.20, self.height * 0.8),
            (self.width * 0.60, self.height * 0.4),
            (self.width * 0.70, self.height * 0.8),
        ]

        self.boxes = []
        self.stop_specific_mode = False
        self.move_specific_mode = False

        # Load background image (if available)
        self.load_background()

        self.create_widgets()
        self.load_images()

        # Video recording setup
        self.init_video_writer()

        # Start animation loop
        self.update_animation()

        # Bind window close event to release video writer
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_background(self):
        """Load the background image, resize to canvas size, and store as PIL and PhotoImage."""
        if os.path.exists(BACKGROUND_IMAGE_PATH):
            try:
                bg = Image.open(BACKGROUND_IMAGE_PATH).convert("RGB")
                bg = bg.resize((self.width, self.height), Image.Resampling.LANCZOS)
                self.bg_image = bg
                self.bg_photo = ImageTk.PhotoImage(bg)
            except Exception as e:
                print(f"Could not load background image: {e}")
                self.bg_image = None
                self.bg_photo = None
        else:
            print(f"Background image '{BACKGROUND_IMAGE_PATH}' not found. Using gray background.")
            self.bg_image = None
            self.bg_photo = None

    def init_video_writer(self):
        """Initialize OpenCV VideoWriter for 'simulated_video.mp4'."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20  # matches the animation update interval (50 ms)
        self.video_writer = cv2.VideoWriter('simulated_video.mp4', fourcc, fps,
                                            (self.width, self.height))

    def create_widgets(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_send = tk.Button(control_frame, text="Send Box", command=self.add_box)
        self.btn_send.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_stop_all = tk.Button(control_frame, text="Stop All", command=self.stop_all_boxes)
        self.btn_stop_all.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_stop_specific = tk.Button(control_frame, text="Stop Specific",
                                           command=self.toggle_stop_specific)
        self.btn_stop_specific.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_move_specific = tk.Button(control_frame, text="Move Specific",
                                           command=self.toggle_move_specific)
        self.btn_move_specific.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas – background color will be overridden by the image if available
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height,
                                bg='light gray')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # If we have a background image, place it on the canvas and lower it
        if self.bg_photo:
            self.bg_canvas_id = self.canvas.create_image(self.width//2, self.height//2,
                                                         image=self.bg_photo)
            self.canvas.tag_lower(self.bg_canvas_id)   # send to back

        # Visual aids (drawn on top of background)
        self.canvas.create_line(0, self.threshold_y, self.width, self.threshold_y,
                                fill='red', dash=(4, 2))
        for i, (sx, sy) in enumerate(self.slots):
            color = 'blue' if i < 2 else 'green'
            self.canvas.create_oval(sx-5, sy-5, sx+5, sy+5, fill=color)

    def load_images(self):
        self.image_paths = []
        image_folder = "boxes"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        if not os.listdir(image_folder):
            for i in range(1, 6):
                img = Image.new('RGB', (100, 100),
                                color=(random.randint(0,255),
                                       random.randint(0,255),
                                       random.randint(0,255)))
                img.save(os.path.join(image_folder, f"dummy_{i}.png"))
        for file in os.listdir(image_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.image_paths.append(os.path.join(image_folder, file))

    def add_box(self):
        if not self.image_paths:
            print("No images available.")
            return
        path = random.choice(self.image_paths)
        x = self.width / 2
        y = 10
        box = Box(self.canvas, x, y, path, self)
        self.boxes.append(box)

    def stop_all_boxes(self):
        for box in self.boxes:
            box.state = 'stopped'

    def toggle_stop_specific(self):
        self.stop_specific_mode = not self.stop_specific_mode
        self.move_specific_mode = False
        self.btn_stop_specific.config(relief=tk.SUNKEN if self.stop_specific_mode else tk.RAISED)
        self.btn_move_specific.config(relief=tk.RAISED)

    def toggle_move_specific(self):
        self.move_specific_mode = not self.move_specific_mode
        self.stop_specific_mode = False
        self.btn_move_specific.config(relief=tk.SUNKEN if self.move_specific_mode else tk.RAISED)
        self.btn_stop_specific.config(relief=tk.RAISED)

    def on_canvas_click(self, event):
        clicked_box = None
        for box in reversed(self.boxes):
            if box.contains_point(event.x, event.y):
                clicked_box = box
                break
        if clicked_box is None:
            return

        if self.stop_specific_mode:
            clicked_box.state = 'stopped'

        elif self.move_specific_mode:
            if (clicked_box.state == 'stopped' and
                not clicked_box.reached_slot):
                clicked_box.state = 'moving'
                if clicked_box.assigned_slot is not None:
                    dx = clicked_box.target_x - clicked_box.x
                    dy = clicked_box.target_y - clicked_box.y
                    clicked_box.vx = dx * 0.1
                    clicked_box.vy = dy * 0.1
                else:
                    clicked_box.vx = 0
                    clicked_box.vy = 2

    def render_frame(self):
        """Create a PIL image of the current simulation state."""
        # Start with background (if available) or light gray
        if self.bg_image:
            img = self.bg_image.copy()
        else:
            img = Image.new('RGB', (self.width, self.height), color='lightgray')
        draw = ImageDraw.Draw(img)

        # Draw red threshold line
        draw.line((0, self.threshold_y, self.width, self.threshold_y),
                  fill='red', width=2)

        # Draw slot circles
        for i, (sx, sy) in enumerate(self.slots):
            color = 'blue' if i < 2 else 'green'
            draw.ellipse((sx-5, sy-5, sx+5, sy+5), fill=color)

        # Draw each box
        for box in self.boxes:
            # Get current box image (already resized)
            scale = 0.5 + (box.y / self.height) * 1.0
            w = max(10, int(box.base_width * scale))
            h = max(10, int(box.base_height * scale))
            box_img = box.original_image.resize((w, h), Image.Resampling.LANCZOS)
            # Paste with alpha if RGBA
            if box_img.mode == 'RGBA':
                img.paste(box_img, (int(box.x - w//2), int(box.y - h//2)), box_img)
            else:
                img.paste(box_img, (int(box.x - w//2), int(box.y - h//2)))

        return img

    def update_animation(self):
        """Update all boxes, then render and record a video frame."""
        for box in self.boxes[:]:
            box.update_position()

        # Render frame and write to video
        try:
            frame_img = self.render_frame()
            frame = np.array(frame_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.video_writer:
                self.video_writer.write(frame)
        except Exception as e:
            print(f"Video frame error: {e}")

        self.root.after(50, self.update_animation)

    def on_closing(self):
        """Release video writer and close window."""
        if self.video_writer:
            self.video_writer.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()