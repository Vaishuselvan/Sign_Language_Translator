import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import pyttsx3
import threading
import PIL.Image, PIL.ImageTk  # Need to install pillow: pip install pillow

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
               27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
               32: 'You are welcome.'}

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Sample demo words
demo_words = {
    "Animals": ["CAT", "DOG", "BIRD", "FISH", "LION"],
    "Greetings": ["HELLO", "THANK YOU", "SORRY", "PLEASE"],
    "Colors": ["RED", "BLUE", "GREEN", "YELLOW", "BLACK"],
    "Food": ["PIZZA", "APPLE", "BREAD", "WATER", "CAKE"],
    "Objects": ["BOOK", "CHAIR", "TABLE", "PHONE", "DOOR"]
}

class SignLanguageApp:
    def __init__(self):
        # Initialize main variables
        self.cap = cv2.VideoCapture(0)
        self.current_letters = []
        self.target_word = ""
        self.current_category = ""
        self.video_running = True
        self.showing_target = False

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Sign Language Translator")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Apply a modern theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Arial", 12), padding=5)
        style.configure("TButton", font=("Arial", 12), padding=5)
        style.configure("TCombobox", font=("Arial", 12), padding=5)
        style.configure("TLabelFrame", font=("Arial", 12, "bold"), padding=10)

        # Create main layout frames
        self.top_frame = ttk.Frame(self.root, padding=10)
        self.top_frame.pack(fill=tk.X)

        self.middle_frame = ttk.Frame(self.root, padding=10)
        self.middle_frame.pack(fill=tk.X)

        self.bottom_frame = ttk.Frame(self.root, padding=10)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Category selection dropdown
        ttk.Label(self.top_frame, text="Select Category:").pack(side=tk.LEFT, padx=5)
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(self.top_frame,
                                             textvariable=self.category_var,
                                             values=list(demo_words.keys()),
                                             state="readonly",
                                             width=15)
        self.category_dropdown.pack(side=tk.LEFT, padx=5)
        self.category_dropdown.bind("<<ComboboxSelected>>", self.on_category_selected)

        # Word selection dropdown
        ttk.Label(self.top_frame, text="Select Word:").pack(side=tk.LEFT, padx=5)
        self.word_var = tk.StringVar()
        self.word_dropdown = ttk.Combobox(self.top_frame,
                                         textvariable=self.word_var,
                                         state="readonly",
                                         width=15)
        self.word_dropdown.pack(side=tk.LEFT, padx=5)
        self.word_dropdown.bind("<<ComboboxSelected>>", self.on_word_selected)

        # Custom input field
        ttk.Label(self.top_frame, text="or Enter Custom Word:").pack(side=tk.LEFT, padx=5)
        self.custom_word_var = tk.StringVar()
        self.custom_word_entry = ttk.Entry(self.top_frame, textvariable=self.custom_word_var, width=15)
        self.custom_word_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.top_frame, text="Set", command=self.set_custom_word).pack(side=tk.LEFT, padx=5)

        # Status display in the middle frame
        self.status_frame = ttk.LabelFrame(self.middle_frame, text="Status", padding=10)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)

        # Target word display
        ttk.Label(self.status_frame, text="Target Word:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.target_label = ttk.Label(self.status_frame, text="", font=("Arial", 14, "bold"))
        self.target_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Current progress display
        ttk.Label(self.status_frame, text="Current Progress:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.progress_label = ttk.Label(self.status_frame, text="", font=("Arial", 14))
        self.progress_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        # Last detected letter
        ttk.Label(self.status_frame, text="Last Detected:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.last_detected_label = ttk.Label(self.status_frame, text="", font=("Arial", 14))
        self.last_detected_label.grid(row=2, column=1, sticky=tk.W, padx=5)

        # Action buttons
        self.button_frame = ttk.Frame(self.middle_frame, padding=10)
        self.button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(self.button_frame, text="Speak Word", command=self.speak_current_word).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Reset", command=self.reset_progress).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Toggle Target", command=self.toggle_target).pack(side=tk.LEFT, padx=5)

        # Camera frame for displaying video
        self.camera_frame = ttk.LabelFrame(self.bottom_frame, text="Camera Feed", padding=10)
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Label to display camera feed
        self.video_label = ttk.Label(self.camera_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Create a canvas for displaying the camera feed
        self.canvas_width = 640
        self.canvas_height = 480
        self.canvas = tk.Canvas(self.camera_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # DEXA name and animation frame
        self.dexa_frame = ttk.LabelFrame(self.bottom_frame, text="DEXA", padding=10)
        self.dexa_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.dexa_canvas = tk.Canvas(self.dexa_frame, width=400, height=400, bg="white")
        self.dexa_canvas.pack(fill=tk.BOTH, expand=True)

        # Start the video processing thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Start the DEXA animation thread
        self.animation_thread = threading.Thread(target=self.animate_dexa)
        self.animation_thread.daemon = True
        self.animation_thread.start()

        # Start the main loop
        self.root.mainloop()

    def on_category_selected(self, event):
        """Handle category selection"""
        self.current_category = self.category_var.get()
        self.word_dropdown['values'] = demo_words[self.current_category]
        self.word_dropdown.current(0)
        self.on_word_selected(None)

    def on_word_selected(self, event):
        """Handle word selection"""
        if self.word_var.get():
            self.target_word = self.word_var.get()
            self.update_target_display()
            self.reset_progress()

    def set_custom_word(self):
        """Set a custom target word"""
        custom_word = self.custom_word_var.get().upper()
        if custom_word and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in custom_word):
            self.target_word = custom_word
            self.update_target_display()
            self.reset_progress()
        else:
            # Show error message for invalid word
            error_window = tk.Toplevel(self.root)
            error_window.title("Error")
            ttk.Label(error_window, text="Please enter a valid word (letters A-Z only)",
                     padding=20).pack()
            error_window.after(2000, error_window.destroy)

    def update_target_display(self):
        """Update the target word display"""
        self.target_label.config(text=self.target_word)
        self.update_progress_display()

    def update_progress_display(self):
        """Update the progress display"""
        current_text = ''.join(self.current_letters)
        self.progress_label.config(text=current_text)

        # Check if word is complete
        if current_text == self.target_word and current_text:
            self.speak_current_word()
            self.show_completion_message()

    def update_last_detected(self, letter):
        """Update the last detected letter display"""
        self.last_detected_label.config(text=letter)

    def reset_progress(self):
        """Reset the current progress"""
        self.current_letters = []
        self.update_progress_display()

    def speak_current_word(self):
        """Speak the current progress using text-to-speech"""
        word = ''.join(self.current_letters)
        if word:
            threading.Thread(target=lambda: engine.say(word) or engine.runAndWait()).start()

    def toggle_target(self):
        """Toggle showing the target letters to sign"""
        self.showing_target = not self.showing_target

    def show_completion_message(self):
        """Show a completion message"""
        completion_window = tk.Toplevel(self.root)
        completion_window.title("Word Complete!")
        ttk.Label(completion_window, text=f"Great job! You completed the word '{self.target_word}'",
                 font=("Arial", 14, "bold"), padding=20).pack()
        completion_window.after(3000, completion_window.destroy)

    def update_frame(self, frame):
        """Update the frame in the Tkinter window"""
        # Convert the frame to a format Tkinter can display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)

        # Resize if needed to fit the canvas
        img = img.resize((self.canvas_width, self.canvas_height), PIL.Image.LANCZOS)

        # Convert to PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image=img)

        # Update canvas with new image
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Add overlays to canvas if needed
        current_text = ''.join(self.current_letters)
        if current_text:
            self.canvas.create_text(10, self.canvas_height - 20, text=f"Current: {current_text}",
                                   fill="red", font=("Arial", 15), anchor=tk.W)

        if self.showing_target and self.target_word:
            if len(self.current_letters) < len(self.target_word):
                target_letter = self.target_word[len(self.current_letters)]
                self.canvas.create_text(10, 50, text=f"Sign: {target_letter}",
                                       fill="green", font=("Arial", 20), anchor=tk.W)

    def process_video(self):
        """Process video frames to detect hand signs"""
        last_prediction = None
        prediction_count = 0
        required_consistent_frames = 10  # Number of consistent frames required to register a letter

        while self.video_running:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = self.cap.read()
            if not ret:
                continue

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Check if prediction is consistent
                    if predicted_character == last_prediction:
                        prediction_count += 1
                    else:
                        last_prediction = predicted_character
                        prediction_count = 1

                    # Update the UI with the current prediction
                    self.root.after(0, self.update_last_detected, predicted_character)

                    # If prediction is consistent for enough frames, add the letter
                    if prediction_count >= required_consistent_frames and len(predicted_character) == 1:
                        # Only add letter if it's the next expected letter or no target word
                        if (not self.target_word or
                            (len(self.current_letters) < len(self.target_word) and
                             predicted_character == self.target_word[len(self.current_letters)])):

                            self.current_letters.append(predicted_character)
                            self.root.after(0, self.update_progress_display)
                            prediction_count = 0  # Reset after adding a letter

                    # Display the prediction on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                except Exception as e:
                    pass

            # Update the Tkinter UI with the new frame
            self.root.after(1, self.update_frame, frame)

            # Short sleep to reduce CPU usage
            import time
            time.sleep(0.01)

    def animate_dexa(self):
        """Animate the DEXA name"""
        angle = 0
        while self.video_running:
            self.dexa_canvas.delete("all")
            self.dexa_canvas.create_text(200, 200, text="DEXA", font=("Arial", 40, "bold"), fill="blue")
            self.dexa_canvas.create_arc(50, 50, 350, 350, start=angle, extent=180, style=tk.ARC, outline="blue", width=5)
            angle = (angle + 10) % 360
            self.dexa_canvas.update()
            time.sleep(0.1)

    def on_closing(self):
        """Handle window closing"""
        self.video_running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = SignLanguageApp()
