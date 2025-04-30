import os
import requests
import logging
from dotenv import load_dotenv
import datetime
import json
import math
import random
import re
import smtplib
import threading
import time
import tkinter as tk
import webbrowser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tkinter import scrolledtext, messagebox, filedialog, Toplevel, Label, Button, Frame, BOTH, X, W, LEFT, END, NORMAL, DISABLED, WORD, StringVar, ttk
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import tensorflow as tf
import torch
from PIL import Image, ImageTk
from bs4 import BeautifulSoup
from deepface import DeepFace
from googleapiclient.discovery import build
from googletrans import Translator
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from google.cloud import vision
from google.cloud.vision_v1 import types
import retrying

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

class PersonalAssistant:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.name = "SWASTI"
        self.owner = "User"
        self.wake_word = "swasti"
        self.listening_for_wake_word = True
        self.root = None
        self.status_var = None
        self.status_label = None
        self.command_entry = None
        self.chat_display = None
        self.theme = "light"
        self.tasks = []
        self.camera_frame = None
        self.video_frame = None
        self.video_label = None
        self.cap = None
        self.is_camera_active = False
        self.current_mode = None
        self.stop_camera_thread = False
        self.progress_bar = None

        # API keys from environment variables
        self.youtube_api_key = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyBUf7FPQmxnBdQjT8MjkNdgu9lQXmmrgBI')
        self.google_api_key = os.environ.get('GOOGLE_API_KEY', 'AIzaSyA2U1rWCgJmXEleN3gBJns-dBc23eDGKHs')
        self.google_cse_id = os.environ.get('GOOGLE_CSE_ID', 'AIzaSyDo1oBkRp-_iZWkNB3ld0FaAk-_vnkbnh8')
        self.weather_api_key = os.environ.get('WEATHER_API_KEY', '06ce023c0be29621fc47e5dee774329b')
        self.google_ai_api_key = os.environ.get('GOOGLE_AI_API_KEY', 'AIzaSyD_xucOvy13oCvH9isblz5L7kWGzvZvOMQ')

        # Check API keys
        for key_name in ['YOUTUBE_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_CSE_ID', 'WEATHER_API_KEY', 'GOOGLE_AI_API_KEY']:
            if not os.environ.get(key_name):
                logging.warning(f"{key_name} not set")
                self.speak(f"{key_name} is not set. Some features may not work.")

        # Google Cloud Vision client
        try:
            self.vision_client = vision.ImageAnnotatorClient()
        except Exception as e:
            logging.error(f"Failed to initialize Google Cloud Vision client: {str(e)}")
            self.vision_client = None
            self.speak("Google Cloud Vision API not initialized. Image analysis may be limited.")

        # AI models
        self.yolo_model = None
        self.crop_model = None
        self.crop_classes = None

        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

        # Translation
        self.translator = Translator()

        # Sympy for calculations
        self.x, self.y = sp.symbols('x y')  # Support multiple variables
        self.transformations = (standard_transformations + (implicit_multiplication_application,))

        # Load tasks from file
        self.load_tasks()

        self.load_models()

    def load_tasks(self):
        """Load tasks from a JSON file."""
        try:
            with open('tasks.json', 'r') as f:
                self.tasks = json.load(f)
        except FileNotFoundError:
            self.tasks = []

    def save_tasks(self):
        """Save tasks to a JSON file."""
        with open('tasks.json', 'w') as f:
            json.dump(self.tasks, f)

    def load_models(self):
        """Load AI models for vision tasks with progress bar."""
        try:
            self.update_status("Loading YOLO model...")
            self.show_progress_bar()
            logging.info("Loading YOLOv5m model")
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True, force_reload=False)
            self.yolo_model.conf = 0.5
            logging.info("YOLO model loaded")

            self.update_status("Loading crop disease model...")
            crop_model_path = 'crop_disease_model.h5'
            if os.path.exists(crop_model_path):
                self.crop_model = tf.keras.models.load_model(crop_model_path)
                self.crop_classes = self.load_crop_classes()
                logging.info("Crop disease model loaded")
            else:
                logging.warning(f"Crop disease model not found at {crop_model_path}")
                self.update_status("Crop disease model not found. Feature disabled.")
                self.speak("Crop disease model not found. This feature is disabled.")
            self.update_status("Models loaded successfully!")
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.speak("Failed to load AI models.")
            self.yolo_model = None
            self.crop_model = None
        finally:
            self.hide_progress_bar()

    def load_crop_classes(self):
        return [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot',
            'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'
        ]

    def show_progress_bar(self):
        """Show a progress bar in the GUI."""
        if self.root:
            self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
            self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
            self.progress_bar.start()
            self.root.update_idletasks()

    def hide_progress_bar(self):
        """Hide the progress bar."""
        if self.progress_bar:
            self.progress_bar.stop()
            self.progress_bar.destroy()
            self.progress_bar = None
            self.root.update_idletasks()

    def speak(self, text):
        """Convert text to speech."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS error: {str(e)}")

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        if self.theme == "light":
            self.theme = "dark"
            self.root.configure(bg="#2b2b2b")
            self.chat_display.configure(bg="#333333", fg="#ffffff")
            self.command_entry.configure(bg="#333333", fg="#ffffff", insertbackground="#ffffff")
            self.status_label.configure(bg="#2b2b2b", fg="#ffffff")
            self.update_chat_display("Switched to dark theme.")
            self.speak("Switched to dark theme.")
        else:
            self.theme = "light"
            self.root.configure(bg="SystemButtonFace")
            self.chat_display.configure(bg="white", fg="black")
            self.command_entry.configure(bg="white", fg="black", insertbackground="black")
            self.status_label.configure(bg="SystemButtonFace", fg="black")
            self.update_chat_display("Switched to light theme.")
            self.speak("Switched to light theme.")

    def create_gui(self):
        """Create the main GUI window."""
        self.root = tk.Tk()
        self.root.title(f"{self.name} - Personal Assistant")
        self.root.geometry("600x500")

        # Title
        title_frame = tk.Frame(self.root, pady=5)
        title_frame.pack(fill=tk.X)
        title_label = tk.Label(title_frame, text=f"{self.name} - Your Assistant", font=("Arial", 16, "bold"))
        title_label.pack()

        # Chat display
        chat_frame = tk.Frame(self.root)
        chat_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
        self.chat_display.pack(expand=True, fill=tk.BOTH)

        # Input
        input_frame = tk.Frame(self.root, pady=5)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        self.command_entry = tk.Entry(input_frame, font=("Arial", 10))
        self.command_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.command_entry.bind("<Return>", lambda event: self.process_gui_command())
        tk.Button(input_frame, text="Send", command=self.process_gui_command).pack(side=tk.LEFT)
        tk.Button(input_frame, text="🎤", command=self.activate_listening).pack(side=tk.LEFT, padx=5)

        # Status
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)

        # Menu
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Change Wake Word", command=self.change_wake_word)
        file_menu.add_command(label="System Status", command=self.show_system_status)
        file_menu.add_command(label="Toggle Theme", command=self.toggle_theme)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_program)
        menu_bar.add_cascade(label="File", menu=file_menu)

        actions_menu = tk.Menu(menu_bar, tearoff=0)
        actions_menu.add_command(label="Set Reminder", command=self.handle_reminder)
        actions_menu.add_command(label="Set Alarm", command=self.handle_alarm)
        actions_menu.add_command(label="Weather", command=self.handle_weather)
        actions_menu.add_command(label="Translate Text", command=self.handle_translation)
        actions_menu.add_command(label="Send Email", command=self.handle_email)
        actions_menu.add_command(label="Manage Tasks", command=self.handle_tasks)
        actions_menu.add_command(label="Calculate", command=self.handle_calculation_dialog)
        menu_bar.add_cascade(label="Actions", menu=actions_menu)

        vision_menu = tk.Menu(menu_bar, tearoff=0)
        vision_menu.add_command(label="Face Detection", command=lambda: self.toggle_camera('face'))
        vision_menu.add_command(label="Object Detection", command=lambda: self.toggle_camera('object'))
        if self.crop_model:
            vision_menu.add_command(label="Crop Disease Detection", command=lambda: self.toggle_camera('crop'))
        vision_menu.add_command(label="Cloud Vision Analysis", command=self.handle_cloud_vision_dialog)
        vision_menu.add_command(label="Stop Camera", command=self.stop_camera)
        vision_menu.add_command(label="Capture Image", command=self.capture_image)
        vision_menu.add_command(label="Upload Image", command=self.upload_image)
        menu_bar.add_cascade(label="Vision", menu=vision_menu)

        ai_menu = tk.Menu(menu_bar, tearoff=0)
        ai_menu.add_command(label="Search YouTube", command=self.search_youtube_dialog)
        ai_menu.add_command(label="Search Google", command=self.search_google_dialog)
        ai_menu.add_command(label="Analyze Article", command=self.analyze_article_dialog)
        ai_menu.add_command(label="Use Google AI Studio", command=self.use_google_ai_studio)
        ai_menu.add_command(label="Open AI Studio Live", command=self.open_ai_studio_live)
        ai_menu.add_command(label="Open AI Studio New Chat", command=self.open_ai_studio_new_chat)
        menu_bar.add_cascade(label="AI Agent", menu=ai_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", f"{self.name} - Personal Assistant v1.2\n\nDeveloped by Motalib\n"
                     f"Enhanced with Google Cloud Vision API, improved object detection, accurate weather updates, "
                     f"translation, email functionality, advanced task management, Google AI Studio integration, "
                     f"and advanced calculation features.\nFor {self.owner}"))
        help_menu.add_command(label="Commands List", command=self.show_commands_list)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)
        self.update_chat_display(f"Hello! I am {self.name}, your assistant. How can I help you today?")
        self.speak(f"Hello! I am {self.name}, your assistant.")
        self.start_wake_word_listener()
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)
        return self.root

    def update_chat_display(self, message: str):
        """Update the chat display with a timestamped message."""
        if self.chat_display:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.insert(tk.END, f"[{timestamp}] {message}\n")
            self.chat_display.see(tk.END)
            self.chat_display.config(state=tk.DISABLED)

    def update_status(self, status: str):
        if self.status_var:
            self.status_var.set(status)
            if self.root:
                self.root.update_idletasks()

    def process_gui_command(self):
        command = self.command_entry.get().strip()
        if command:
            self.update_chat_display(f"You: {command}")
            self.command_entry.delete(0, tk.END)
            self.handle_command(command.lower())

    def activate_listening(self):
        self.update_status("Listening...")
        threading.Thread(target=self.listen_and_process_command, daemon=True).start()

    def listen_and_process_command(self):
        query = self.take_command()
        if query != "None":
            self.handle_command(query)

    def take_command(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.update_status("Listening for command...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                self.update_status(f"Command received: {command}")
                logging.info(f"Recognized: {command}")
                return command
            except sr.WaitTimeoutError:
                self.update_status("No command detected")
                return "None"
            except sr.UnknownValueError:
                self.update_status("Could not understand audio")
                return "None"
            except sr.RequestError as e:
                self.update_status(f"Speech error: {e}")
                logging.error(f"Speech error: {e}")
                return "None"

    def start_wake_word_listener(self):
        def wake_word_loop():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                while self.listening_for_wake_word:
                    try:
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        text = recognizer.recognize_google(audio).lower()
                        if self.wake_word in text:
                            self.update_status(f"Wake word '{self.wake_word}' detected")
                            self.listen_and_process_command()
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        continue
                    except sr.RequestError as e:
                        logging.error(f"Wake word error: {e}")
                    time.sleep(0.1)
        threading.Thread(target=wake_word_loop, daemon=True).start()

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def use_google_ai_studio(self):
        """Interact with Google Gemini API with retry logic."""
        self.update_chat_display("Interacting with Google Gemini API...")
        self.speak("Interacting with Google Gemini API...")
        self.show_progress_bar()

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.google_ai_api_key}
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": "Provide a brief summary of artificial intelligence in 100 words or less."}
                    ]
                }
            ]
        }

        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response")
            self.update_chat_display(f"Response from Google Gemini API: {generated_text}")
            self.speak(f"Response: {generated_text}")
        except requests.RequestException as e:
            error_message = f"Error interacting with Google Gemini API: {str(e)}"
            self.update_chat_display(error_message)
            self.speak("Error interacting with Google Gemini API.")
            logging.error(error_message)
            raise
        finally:
            self.hide_progress_bar()

    def open_ai_studio_live(self):
        """Open Google AI Studio Live page."""
        url = "https://aistudio.google.com/live"
        webbrowser.open(url)
        self.update_chat_display("Opening Google AI Studio Live...")
        self.speak("Opening Google AI Studio Live")

    def open_ai_studio_new_chat(self):
        """Open Google AI Studio New Chat page."""
        url = "https://aistudio.google.com/prompts/new_chat"
        webbrowser.open(url)
        self.update_chat_display("Opening Google AI Studio New Chat...")
        self.speak("Opening Google AI Studio New Chat")

    def safe_calculate(self, expression):
        """Safely evaluate a mathematical expression using sympy with multiple variables."""
        try:
            # Parse the expression with sympy
            expr = parse_expr(expression, transformations=self.transformations, local_dict={'x': self.x, 'y': self.y})
            # If it's a numerical expression, evaluate it
            if expr.is_number:
                result = float(expr.evalf())
                return f"{result:.4f}" if result % 1 != 0 else str(int(result))
            # If it's an algebraic expression, simplify it
            simplified = sp.simplify(expr)
            return str(simplified)
        except Exception as e:
            logging.error(f"Calculation error: {str(e)}")
            return f"Error calculating: {str(e)}"

    def handle_calculation_dialog(self):
        """Open a dialog for performing calculations."""
        dialog = Toplevel(self.root)
        dialog.title("Calculator")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = Frame(dialog, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        Label(frame, text="Enter Expression (e.g., 2+3, sin(pi/2), x^2+2*x+y):").pack(anchor=tk.W, pady=(0, 5))
        expr_entry = tk.Entry(frame, width=40)
        expr_entry.pack(fill=tk.X, pady=(0, 10))
        expr_entry.focus_set()

        result_label = Label(frame, text="Result: ", wraplength=350)
        result_label.pack(anchor=tk.W, pady=(0, 10))

        def calculate():
            expr = expr_entry.get().strip()
            if expr:
                result = self.safe_calculate(expr)
                result_label.config(text=f"Result: {result}")
                self.update_chat_display(f"Calculated: {expr} = {result}")
                self.speak(f"Result: {result}")
            else:
                messagebox.showwarning("Invalid Input", "Please enter a mathematical expression.")
                self.speak("Please enter an expression.")

        Button(frame, text="Calculate", command=calculate).pack(pady=5)
        expr_entry.bind("<Return>", lambda event: calculate())

        # Example expressions
        Label(frame, text="Examples: 2+3*4, sin(pi/2), x^2+2*x+y, log(10)").pack(anchor=tk.W, pady=(10, 0))

    def handle_command(self, command):
        """Process user commands."""
        self.update_chat_display(f"{self.name}: Processing command: {command}")
        try:
            if command.startswith("search youtube for "):
                query = command.replace("search youtube for ", "").strip()
                self.search_youtube(query)
            elif command.startswith("search google for "):
                query = command.replace("search google for ", "").strip()
                self.search_google(query)
            elif command.startswith("analyze article "):
                url = command.replace("analyze article ", "").strip()
                self.analyze_article(url)
            elif command == "use google ai studio":
                self.use_google_ai_studio()
            elif command == "open ai studio live":
                self.open_ai_studio_live()
            elif command == "open ai studio new chat":
                self.open_ai_studio_new_chat()
            elif command == "face detection":
                self.toggle_camera('face')
            elif command == "object detection":
                self.toggle_camera('object')
            elif command == "crop disease detection":
                if self.crop_model:
                    self.toggle_camera('crop')
                else:
                    self.update_chat_display("Crop disease detection unavailable.")
                    self.speak("Crop disease detection unavailable.")
            elif command == "cloud vision analysis":
                self.handle_cloud_vision_dialog()
            elif command == "stop camera":
                self.stop_camera()
            elif command == "open youtube":
                webbrowser.open("https://www.youtube.com")
                self.update_chat_display("Opening YouTube...")
                self.speak("Opening YouTube")
            elif command == "open google":
                webbrowser.open("https://www.google.com")
                self.update_chat_display("Opening Google...")
                self.speak("Opening Google")
            elif command in ("open claude", "open chatgpt", "open copilot"):
                self.update_chat_display(f"Sorry, I cannot open {command.split('open ')[1].title()} directly. Please visit their website.")
                self.speak(f"Sorry, I cannot open {command.split('open ')[1].title()} directly.")
            elif command == "open whatsapp":
                webbrowser.open("https://web.whatsapp.com")
                self.update_chat_display("Opening WhatsApp...")
                self.speak("Opening WhatsApp")
            elif command == "open vs code":
                try:
                    os.system("code")
                    self.update_chat_display("Opening Visual Studio Code...")
                    self.speak("Opening Visual Studio Code")
                except:
                    self.update_chat_display("Error opening Visual Studio Code. Please ensure it is installed.")
                    self.speak("Error opening Visual Studio Code")
            elif command in ("what time is it", "current time"):
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                self.update_chat_display(f"The current time is {current_time}")
                self.speak(f"The current time is {current_time}")
            elif command in ("what's the date", "current date"):
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                self.update_chat_display(f"Today's date is {current_date}")
                self.speak(f"Today's date is {current_date}")
            elif command.startswith("weather in "):
                city = command.replace("weather in ", "").strip()
                self.get_weather(city)
            elif command == "set reminder":
                self.handle_reminder()
            elif command == "set alarm":
                self.handle_alarm()
            elif command == "tell me a joke":
                jokes = [
                    "Why did the scarecrow become a motivational speaker? Because he was outstanding in his field!",
                    "Why can't programmers prefer dark mode? Because the light attracts bugs.",
                    "What do you call a dinosaur that takes care of its teeth? A Flossiraptor!"
                ]
                joke = random.choice(jokes)
                self.update_chat_display(joke)
                self.speak(joke)
            elif command in ("good morning", "good evening", "good night"):
                self.update_chat_display(f"{command.title()}! How can I assist you today?")
                self.speak(f"{command.title()}! How can I assist you today?")
            elif command == "who are you":
                self.update_chat_display(f"I am {self.name}, your personal assistant created to help with tasks, searches, and more!")
                self.speak(f"I am {self.name}, your personal assistant created to help with tasks, searches, and more!")
            elif command.startswith("translate "):
                parts = command.split(" to ")
                if len(parts) == 2:
                    text = parts[0].replace("translate ", "").strip()
                    language = parts[1].strip()
                    self.translate_text(text, language)
                else:
                    self.update_chat_display("Please use format 'translate [text] to [language]'")
                    self.speak("Please use format 'translate text to language'")
            elif command == "send email":
                self.handle_email()
            elif command == "change wake word":
                self.change_wake_word()
            elif command == "system status":
                self.show_system_status()
            elif command.startswith("calculate "):
                expression = command.replace("calculate ", "").strip()
                result = self.safe_calculate(expression)
                self.update_chat_display(f"Result: {result}")
                self.speak(f"The result is {result}")
            elif command == "toggle theme":
                self.toggle_theme()
            elif command == "manage tasks":
                self.handle_tasks()
            elif command in ("exit", "quit", "goodbye"):
                self.exit_program()
            else:
                self.update_chat_display(f"I don’t understand '{command}'. Searching Google...")
                self.speak("I don’t understand that command. Searching Google.")
                self.search_google(command)
            return True
        except Exception as e:
            self.update_chat_display(f"Error: {str(e)}")
            self.speak("An error occurred.")
            logging.error(f"Command error: {str(e)}")
            return False

    def toggle_camera(self, mode):
        """Start camera with specified mode."""
        try:
            if mode == 'crop' and not self.crop_model:
                messagebox.showerror("Error", "Crop disease model not loaded.")
                self.speak("Crop disease model not loaded.")
                return
            if mode == 'object' and not self.yolo_model:
                messagebox.showerror("Error", "YOLO model not loaded.")
                self.speak("YOLO model not loaded.")
                return
            self.stop_camera()
            self.current_mode = mode
            vision_window = Toplevel(self.root)
            vision_window.title(f"{mode.title()} Detection")
            vision_window.geometry("500x400")
            self.camera_frame = Frame(vision_window, bg="black")
            self.camera_frame.pack(expand=True, fill=BOTH, padx=10, pady=5)
            self.video_frame = Frame(self.camera_frame, bg="black", width=480, height=360)
            self.video_frame.pack(expand=True, fill=BOTH)
            self.video_label = Label(self.video_frame, bg="black")
            self.video_label.pack(expand=True, fill=BOTH)
            controls_frame = Frame(vision_window)
            controls_frame.pack(fill=X, pady=5)
            Button(controls_frame, text="Stop Camera", command=self.stop_camera).pack(side=LEFT, padx=5)
            self.is_camera_active = True
            self.stop_camera_thread = False
            threading.Thread(target=self.camera_loop, daemon=True).start()
        except Exception as e:
            logging.error(f"Camera toggle error: {str(e)}")
            self.update_chat_display(f"Error starting camera: {str(e)}")
            self.speak("Error starting camera.")

    def stop_camera(self):
        if self.is_camera_active:
            self.stop_camera_thread = True
            self.is_camera_active = False
            time.sleep(0.5)
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.video_label:
                self.video_label.config(image='')
            if self.camera_frame:
                self.camera_frame.master.destroy()
            self.update_status("Camera stopped")
            self.speak("Camera stopped")

    def camera_loop(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera.")
                self.speak("Could not open camera.")
                return
            while self.is_camera_active and not self.stop_camera_thread:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    break
                if self.current_mode == 'face':
                    frame = self.process_face_detection(frame)
                elif self.current_mode == 'object':
                    frame = self.process_object_detection(frame)
                elif self.current_mode == 'crop':
                    frame = self.process_crop_disease(frame)
                self.display_frame(frame)
                time.sleep(0.03)
        except Exception as e:
            logging.error(f"Camera error: {str(e)}")
            self.update_status(f"Camera error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()

    def display_frame(self, frame):
        if frame is not None and self.video_label:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            width, height = self.video_frame.winfo_width(), self.video_frame.winfo_height()
            if width > 1 and height > 1:
                frame_rgb = cv2.resize(frame_rgb, (width, height))
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk

    def process_face_detection(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            for face in (result if isinstance(result, list) else [result]):
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                emotion = max(face['emotion'], key=face['emotion'].get)
                cv2.putText(frame, f"{emotion}: {face['emotion'][emotion]:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return frame
        except Exception as e:
            logging.error(f"Face detection error: {str(e)}")
            return frame

    def process_object_detection(self, frame):
        if self.yolo_model:
            results = self.yelo_model(frame)
            detections = results.pandas().xyxy[0]
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{detection['name']}: {detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            return frame

    def process_crop_disease(self, frame):
        if self.crop_model:
            img = cv2.resize(frame, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)
            predictions = self.crop_model.predict(img, verbose=0)[0]
            idx = np.argmax(predictions)
            cv2.putText(frame, f"{self.crop_classes[idx]}: {predictions[idx]*100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def capture_image(self):
        """Capture and save an image from the active camera."""
        try:
            if not self.is_camera_active or self.cap is None:
                messagebox.showinfo("Camera", "Camera is not active.")
                self.speak("Camera is not active.")
                return

            for _ in range(3):
                ret, frame = self.cap.read()
                if ret:
                    break
                time.sleep(0.1)
            else:
                messagebox.showerror("Capture Error", "Failed to capture image after multiple attempts.")
                self.speak("Failed to capture image.")
                return

            if self.current_mode == 'face':
                frame = self.process_face_detection(frame)
                analysis_text = "Face detection analysis complete."
            elif self.current_mode == 'object':
                frame = self.process_object_detection(frame)
                analysis_text = "Object detection analysis complete."
            elif self.current_mode == 'crop':
                frame = self.process_crop_disease(frame)
                analysis_text = "Crop disease analysis complete."
            else:
                analysis_text = "Image captured."

            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, frame)
                self.update_chat_display(f"{analysis_text} Image saved to {file_path}")
                self.speak(f"{analysis_text} Image saved.")
        except Exception as e:
            self.update_chat_display(f"Error capturing image: {str(e)}")
            self.speak("Error capturing image.")
            logging.error(f"Error capturing image: {str(e)}")

    def handle_cloud_vision_dialog(self):
        """Open a dialog to select an image for Google Cloud Vision analysis."""
        if not self.vision_client:
            messagebox.showerror("Error", "Google Cloud Vision API not initialized.")
            self.speak("Google Cloud Vision API not initialized.")
            return

        dialog = Toplevel(self.root)
        dialog.title("Cloud Vision Analysis")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = Frame(dialog, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        Label(frame, text="Select an image for analysis:").pack(anchor=tk.W, pady=(0, 10))
        Button(frame, text="Browse", command=lambda: self.analyze_cloud_vision(dialog)).pack(pady=5)
        Label(frame, text="Analysis includes labels, text, and landmarks.").pack(anchor=tk.W, pady=(10, 0))

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def analyze_cloud_vision(self, dialog):
        """Analyze an image using Google Cloud Vision API."""
        dialog.destroy()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        self.update_chat_display("Analyzing image with Google Cloud Vision API...")
        self.speak("Analyzing image with Google Cloud Vision API...")
        self.show_progress_bar()

        try:
            with open(file_path, 'rb') as image_file:
                content = image_file.read()

            image = types.Image(content=content)
            analysis_text = []

            # Label detection
            response = self.vision_client.label_detection(image=image)
            labels = response.label_annotations
            if labels:
                analysis_text.append("Labels:")
                for label in labels[:5]:
                    analysis_text.append(f"- {label.description} (Confidence: {label.score:.2f})")

            # Text detection
            response = self.vision_client.text_detection(image=image)
            texts = response.text_annotations
            if texts:
                analysis_text.append("\nDetected Text:")
                analysis_text.append(texts[0].description[:100] + "..." if len(texts[0].description) > 100 else texts[0].description)

            # Landmark detection
            response = self.vision_client.landmark_detection(image=image)
            landmarks = response.landmark_annotations
            if landmarks:
                analysis_text.append("\nLandmarks:")
                for landmark in landmarks[:3]:
                    analysis_text.append(f"- {landmark.description} (Confidence: {landmark.score:.2f})")

            if not analysis_text:
                analysis_text.append("No features detected in the image.")

            analysis_result = "\n".join(analysis_text)
            self.update_chat_display(f"Cloud Vision Analysis:\n{analysis_result}")
            self.speak("Cloud Vision analysis complete.")
        except Exception as e:
            self.update_chat_display(f"Error analyzing image with Cloud Vision: {str(e)}")
            self.speak("Error analyzing image with Cloud Vision.")
            logging.error(f"Cloud Vision error: {str(e)}")
            raise
        finally:
            self.hide_progress_bar()

    def upload_image(self):
        """Upload and analyze an image with multiple analysis options."""
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
            if not file_path:
                return
            frame = cv2.imread(file_path)
            if frame is None:
                messagebox.showerror("Error", "Could not load the image.")
                self.speak("Could not load the image.")
                return

            dialog = Toplevel(self.root)
            dialog.title("Choose Analysis Type")
            dialog.geometry("400x200")
            dialog.transient(self.root)
            dialog.grab_set()
            frame_dialog = Frame(dialog, padx=10, pady=10)
            frame_dialog.pack(fill=tk.BOTH, expand=True)

            Label(frame_dialog, text="Select analysis type:").pack(anchor=tk.W, pady=(0, 10))
            Button(frame_dialog, text="Crop Disease Analysis", command=lambda: self.process_uploaded_image(dialog, frame, file_path, 'crop')).pack(fill=tk.X, pady=5)
            Button(frame_dialog, text="Object Detection", command=lambda: self.process_uploaded_image(dialog, frame, file_path, 'object')).pack(fill=tk.X, pady=5)
            if self.vision_client:
                Button(frame_dialog, text="Cloud Vision Analysis", command=lambda: self.process_uploaded_image(dialog, frame, file_path, 'cloud')).pack(fill=tk.X, pady=5)

        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing image: {str(e)}")
            self.speak("Error processing image.")
            logging.error(f"Error processing image: {str(e)}")

    def process_uploaded_image(self, dialog, frame, file_path, analysis_type):
        """Process the uploaded image based on the selected analysis type."""
        dialog.destroy()
        if analysis_type == 'crop':
            if self.crop_model is None:
                messagebox.showerror("Error", "Crop disease model not loaded.")
                self.speak("Crop disease model not loaded.")
                return
            frame = self.process_crop_disease(frame)
            analysis_text = "Crop disease analysis complete."
        elif analysis_type == 'object':
            if self.yolo_model is None:
                messagebox.showerror("Error", "YOLO model not loaded.")
                self.speak("YOLO model not loaded.")
                return
            frame = self.process_object_detection(frame)
            analysis_text = "Object detection analysis complete."
        elif analysis_type == 'cloud':
            if self.vision_client is None:
                messagebox.showerror("Error", "Google Cloud Vision API not initialized.")
                self.speak("Google Cloud Vision API not initialized.")
                return
            self.analyze_cloud_vision(file_path=file_path)
            return

        self.update_chat_display(analysis_text)
        self.speak(analysis_text)
        if messagebox.askyesno("Save Result", "Would you like to save the processed image?"):
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if save_path:
                cv2.imwrite(save_path, frame)
                self.update_chat_display(f"Processed image saved to {save_path}")
                self.speak("Processed image saved.")

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def search_youtube(self, query):
        if not self.youtube_api_key:
            self.update_chat_display("YouTube API key not set. Please set YOUTUBE_API_KEY.")
            self.speak("YouTube API key not set.")
            return
        threading.Thread(target=self._search_youtube_thread, args=(query,), daemon=True).start()

    def _search_youtube_thread(self, query):
        self.update_status(f"Searching YouTube for: {query}")
        self.update_chat_display(f"Searching YouTube for '{query}'...")
        self.speak(f"Searching YouTube for {query}")
        self.show_progress_bar()
        try:
            youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            response = youtube.search().list(q=query, part='snippet', maxResults=5, type='video').execute()
            for item in response.get('items', []):
                self.update_chat_display(f"{item['snippet']['title']} - {item['id']['videoId']}")
            self.update_status("YouTube search completed")
        except Exception as e:
            self.update_chat_display(f"YouTube error: {str(e)}")
            self.speak("Error searching YouTube.")
            logging.error(f"YouTube error: {str(e)}")
        finally:
            self.hide_progress_bar()

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def search_google(self, query):
        if not self.google_api_key or not self.google_cse_id:
            self.update_chat_display("Google API keys not set. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID.")
            self.speak("Google API keys not set.")
            return
        threading.Thread(target=self._search_google_thread, args=(query,), daemon=True).start()

    def _search_google_thread(self, query):
        self.update_status(f"Searching Google for: {query}")
        self.update_chat_display(f"Searching Google for '{query}'...")
        self.speak(f"Searching Google for {query}")
        self.show_progress_bar()
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            result = service.cse().list(q=query, cx=self.google_cse_id, num=5).execute()
            for item in result.get('items', []):
                self.update_chat_display(f"{item['title']} - {item['link']}")
            self.update_status("Google search completed")
        except Exception as e:
            self.update_chat_display(f"Google error: {str(e)}")
            self.speak("Error searching Google.")
            logging.error(f"Google error: {str(e)}")
        finally:
            self.hide_progress_bar()

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def analyze_article(self, url):
        threading.Thread(target=self._analyze_article_thread, args=(url,), daemon=True).start()

    def _analyze_article_thread(self, url):
        self.update_status(f"Analyzing article: {url}")
        self.update_chat_display(f"Analyzing article at '{url}'...")
        self.speak(f"Analyzing article at {url}")
        self.show_progress_bar()
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').get_text() if soup.find('title') else "Unknown Title"
            paragraphs = soup.find_all('p')
            main_content = ' '.join([p.get_text().strip() for p in paragraphs])
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            sentences = re.split(r'[.!?]+', main_content)
            summary = '. '.join(sentences[:3]) + '.' if sentences else "No content available."
            self.update_chat_display(f"Article Analysis Complete")
            self.update_chat_display(f"Title: {title}")
            self.update_chat_display(f"Summary: {summary}")
            self.speak(f"Article analysis complete. Title: {title}")
            self.update_status("Article analysis completed")
        except Exception as e:
            self.update_chat_display(f"Error analyzing article: {str(e)}")
            self.speak("Error analyzing article.")
            logging.error(f"Article error: {str(e)}")
        finally:
            self.hide_progress_bar()

    def change_wake_word(self):
        dialog = Toplevel(self.root)
        dialog.title("Change Wake Word")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        Label(dialog, text=f"Current wake word: {self.wake_word}").pack(pady=10)
        Label(dialog, text="Enter new wake word:").pack(pady=5)
        entry = tk.Entry(dialog, width=20)
        entry.pack(pady=5)
        entry.focus_set()
        def submit():
            new_wake_word = entry.get().strip().lower()
            if new_wake_word:
                self.wake_word = new_wake_word
                dialog.destroy()
                self.update_chat_display(f"Wake word changed to '{self.wake_word}'")
                self.speak(f"Wake word changed to {self.wake_word}")
        Button(dialog, text="Change", command=submit).pack(pady=10)
        entry.bind("<Return>", lambda event: submit())

    def show_system_status(self):
        dialog = Toplevel(self.root)
        dialog.title("System Status")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        status_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=40, height=15)
        status_text.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        status_info = [
            f"Assistant Name: {self.name}",
            f"Owner: {self.owner}",
            f"Wake Word: {self.wake_word}",
            f"Theme: {self.theme.capitalize()}",
            f"Camera Status: {'Active' if self.is_camera_active else 'Inactive'}",
            f"Current Camera Mode: {self.current_mode if self.current_mode else 'None'}",
            f"Cloud Vision API: {'Initialized' if self.vision_client else 'Not Initialized'}",
            f"System Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        for info in status_info:
            status_text.insert(tk.END, info + "\n\n")
        status_text.config(state=tk.DISABLED)
        Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def handle_reminder(self):
        dialog = Toplevel(self.root)
        dialog.title("Set Reminder")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = Frame(dialog, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        Label(frame, text="Reminder Message:").pack(anchor=W, pady=(0, 5))
        message_entry = tk.Entry(frame, width=40)
        message_entry.pack(fill=X, pady=(0, 10))
        message_entry.focus_set()
        time_frame = Frame(frame)
        time_frame.pack(fill=X, pady=(0, 10))
        Label(time_frame, text="Time (minutes):").pack(side=LEFT, padx=(0, 5))
        time_entry = tk.Entry(time_frame, width=10)
        time_entry.pack(side=LEFT)
        time_entry.insert(0, "5")
        def submit():
            message = message_entry.get().strip()
            try:
                minutes = int(time_entry.get().strip())
                if message and minutes > 0:
                    dialog.destroy()
                    self.set_reminder(message, minutes)
                else:
                    messagebox.showwarning("Invalid Input", "Please enter a valid message and time.")
                    self.speak("Please enter a valid message and time.")
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid number of minutes.")
                self.speak("Please enter a valid number of minutes.")
        Button(frame, text="Set Reminder", command=submit).pack(pady=10)

    def set_reminder(self, message, minutes):
        self.update_chat_display(f"Setting a reminder for {minutes} minute(s): {message}")
        self.speak(f"Setting a reminder for {minutes} minutes: {message}")
        def reminder_thread():
            time.sleep(minutes * 60)
            self.update_chat_display(f"REMINDER: {message}")
            self.speak(f"Reminder: {message}")
            messagebox.showinfo("Reminder", message)
        threading.Thread(target=reminder_thread, daemon=True).start()

    def handle_alarm(self):
        dialog = Toplevel(self.root)
        dialog.title("Set Alarm")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = Frame(dialog, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        time_frame = Frame(frame)
        time_frame.pack(fill=X, pady=10)
        Label(time_frame, text="Hours:").pack(side=LEFT, padx=(0, 5))
        hours_entry = tk.Entry(time_frame, width=5)
        hours_entry.pack(side=LEFT, padx=(0, 10))
        hours_entry.insert(0, time.strftime("%H"))
        Label(time_frame, text="Minutes:").pack(side=LEFT, padx=(0, 5))
        minutes_entry = tk.Entry(time_frame, width=5)
        minutes_entry.pack(side=LEFT)
        minutes_entry.insert(0, time.strftime("%M"))
        Label(frame, text="Alarm Message (optional):").pack(anchor=W, pady=(0, 5))
        message_entry = tk.Entry(frame, width=40)
        message_entry.pack(fill=X, pady=(0, 10))
        message_entry.focus_set()
        def submit():
            try:
                hours = int(hours_entry.get().strip())
                minutes = int(minutes_entry.get().strip())
                message = message_entry.get().strip() or "Alarm"
                if 0 <= hours < 24 and 0 <= minutes < 60:
                    dialog.destroy()
                    self.set_alarm(hours, minutes, message)
                else:
                    messagebox.showwarning("Invalid Input", "Please enter valid hours (0-23) and minutes (0-59).")
                    self.speak("Please enter valid hours and minutes.")
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter valid numbers for hours and minutes.")
                self.speak("Please enter valid numbers for hours and minutes.")
        Button(frame, text="Set Alarm", command=submit).pack(pady=10)

    def set_alarm(self, hours, minutes, message="Alarm"):
        now = time.localtime()
        alarm_time = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hours, minutes, 0, now.tm_wday, now.tm_yday, now.tm_isdst))
        current_time = time.time()
        if alarm_time <= current_time:
            alarm_time += 24 * 60 * 60
        wait_time = alarm_time - current_time
        alarm_time_str = time.strftime("%H:%M", time.localtime(alarm_time))
        self.update_chat_display(f"Alarm set for {alarm_time_str}: {message}")
        self.speak(f"Alarm set for {alarm_time_str}: {message}")
        def alarm_thread():
            time.sleep(wait_time)
            self.update_chat_display(f"ALARM: {message}")
            self.speak(f"Alarm: {message}")
            messagebox.showinfo("Alarm", message)
        threading.Thread(target=alarm_thread, daemon=True).start()

    def handle_weather(self):
        dialog = Toplevel(self.root)
        dialog.title("Weather Information")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        Label(dialog, text="Enter city name:").pack(pady=10)
        entry = tk.Entry(dialog, width=30)
        entry.pack(pady=10)
        entry.focus_set()
        def submit():
            city = entry.get().strip()
            if city:
                dialog.destroy()
                self.get_weather(city)
        Button(dialog, text="Get Weather", command=submit).pack(pady=10)
        entry.bind("<Return>", lambda event: submit())

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_weather(self, city):
        """Fetch weather information for a given city."""
        self.update_chat_display(f"Fetching weather for {city}...")
        self.speak(f"Fetching weather for {city}")
        self.show_progress_bar()
        if not self.weather_api_key:
            self.update_chat_display("Weather API key not set. Please set WEATHER_API_KEY with a valid OpenWeatherMap API key.")
            self.speak("Weather API key not set.")
            return
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['cod'] == 200:
                temp = data['main']['temp']
                feels_like = data['main']['feels_like']
                humidity = data['main']['humidity']
                description = data['weather'][0]['description'].capitalize()
                city_name = data['name']
                weather_info = (
                    f"Weather in {city_name}:\n"
                    f"Temperature: {temp}°C\n"
                    f"Feels like: {feels_like}°C\n"
                    f"Humidity: {humidity}%\n"
                    f"Description: {description}"
                )
                self.update_chat_display(weather_info)
                self.speak(f"Weather in {city_name}: {temp} degrees Celsius, {description}")
            else:
                self.update_chat_display(f"Could not fetch weather for {city}: {data['message']}")
                self.speak(f"Could not fetch weather for {city}")
            self.update_status("Weather fetch completed")
        except requests.RequestException as e:
            self.update_chat_display(f"Error fetching weather: {str(e)}")
            self.speak("Error fetching weather.")
            logging.error(f"Weather error: {str(e)}")
            raise
        except Exception as e:
            self.update_chat_display(f"Unexpected error fetching weather: {str(e)}")
            self.speak("Unexpected error fetching weather.")
            logging.error(f"Weather error: {str(e)}")
            raise
        finally:
            self.hide_progress_bar()

    def handle_tasks(self):
        """Enhanced task manager with due dates and deletion."""
        dialog = Toplevel(self.root)
        dialog.title("Task Manager")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = Frame(dialog, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        Label(frame, text="Task Manager", font=("Arial", 14, "bold")).pack(pady=10)

        task_frame = Frame(frame)
        task_frame.pack(fill=X, pady=5)
        Label(task_frame, text="Task:").pack(side=LEFT)
        task_entry = tk.Entry(task_frame, width=30)
        task_entry.pack(side=LEFT, padx=5)
        task_entry.focus_set()

        due_frame = Frame(frame)
        due_frame.pack(fill=X, pady=5)
        Label(due_frame, text="Due Date (YYYY-MM-DD):").pack(side=LEFT)
        due_entry = tk.Entry(due_frame, width=15)
        due_entry.pack(side=LEFT, padx=5)
        due_entry.insert(0, datetime.datetime.now().strftime("%Y-%m-%d"))

        task_list = scrolledtext.ScrolledText(frame, wrap=WORD, width=50, height=10)
        task_list.pack(fill=tk.BOTH, expand=True, pady=10)
        task_list.config(state=DISABLED)

        def update_task_list():
            task_list.config(state=NORMAL)
            task_list.delete(1.0, END)
            for i, task in enumerate(self.tasks, 1):
                status = "✔" if task['completed'] else "⬜"
                due_date = task.get('due_date', 'No due date')
                task_list.insert(END, f"{i}. {status} {task['task']} (Due: {due_date})\n")
            task_list.config(state=DISABLED)

        def add_task():
            task = task_entry.get().strip()
            due_date = due_entry.get().strip()
            if task:
                try:
                    if due_date:
                        datetime.datetime.strptime(due_date, "%Y-%m-%d")
                    self.tasks.append({'task': task, 'completed': False, 'due_date': due_date})
                    self.save_tasks()
                    task_entry.delete(0, END)
                    update_task_list()
                    self.update_chat_display(f"Added task: {task}")
                    self.speak(f"Added task: {task}")
                except ValueError:
                    messagebox.showwarning("Invalid Date", "Please enter a valid date (YYYY-MM-DD) or leave blank.")
                    self.speak("Invalid date format.")
            else:
                messagebox.showwarning("Invalid Input", "Please enter a task.")
                self.speak("Please enter a task.")

        def toggle_task():
            try:
                task_num = int(task_entry.get().strip()) - 1
                if 0 <= task_num < len(self.tasks):
                    self.tasks[task_num]['completed'] = not self.tasks[task_num]['completed']
                    self.save_tasks()
                    update_task_list()
                    status = "completed" if self.tasks[task_num]['completed'] else "uncompleted"
                    self.update_chat_display(f"Marked task {task_num + 1} as {status}")
                    self.speak(f"Marked task {task_num + 1} as {status}")
                else:
                    messagebox.showwarning("Invalid Input", "Please enter a valid task number.")
                    self.speak("Please enter a valid task number.")
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid task number.")
                self.speak("Please enter a valid task number.")

        def delete_task():
            try:
                task_num = int(task_entry.get().strip()) - 1
                if 0 <= task_num < len(self.tasks):
                    task = self.tasks.pop(task_num)
                    self.save_tasks()
                    update_task_list()
                    self.update_chat_display(f"Deleted task: {task['task']}")
                    self.speak(f"Deleted task: {task['task']}")
                else:
                    messagebox.showwarning("Invalid Input", "Please enter a valid task number.")
                    self.speak("Please enter a valid task number.")
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid task number.")
                self.speak("Please enter a valid task number.")

        button_frame = Frame(frame)
        button_frame.pack(fill=X, pady=5)
        Button(button_frame, text="Add Task", command=add_task).pack(side=LEFT, padx=5)
        Button(button_frame, text="Toggle Completion", command=toggle_task).pack(side=LEFT, padx=5)
        Button(button_frame, text="Delete Task", command=delete_task).pack(side=LEFT, padx=5)

        update_task_list()

    def handle_translation(self):
        dialog = Toplevel(self.root)
        dialog.title("Text Translation")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = Frame(dialog, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        Label(frame, text="Enter text to translate:").pack(anchor=W, pady=(0, 5))
        text_entry = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=50, height=5)
        text_entry.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        text_entry.focus_set()
        lang_frame = Frame(frame)
        lang_frame.pack(fill=X, pady=(0, 10))
        Label(lang_frame, text="Target Language:").pack(side=LEFT, padx=(0, 5))
        languages = ["Spanish", "French", "German", "Hindi", "Oriya", "English", "Italian", "Portuguese", "Russian", "Japanese", "Chinese", "Arabic"]
        lang_var = StringVar(dialog)
        lang_var.set(languages[0])
        lang_menu = tk.OptionMenu(lang_frame, lang_var, *languages)
        lang_menu.pack(side=LEFT)
        def submit():
            text = text_entry.get("1.0", tk.END).strip()
            language = lang_var.get()
            if text:
                dialog.destroy()
                self.translate_text(text, language)
            else:
                messagebox.showwarning("Invalid Input", "Please enter text to translate.")
                self.speak("Please enter text to translate.")
        Button(frame, text="Translate", command=submit).pack(pady=10)

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def translate_text(self, text, language):
        """Translate text to the target language using googletrans."""
        self.update_chat_display(f"Translating text to {language}...")
        self.speak(f"Translating text to {language}")
        self.show_progress_bar()
        try:
            lang_map = {
                "Spanish": "es", "French": "fr", "German": "de", "Hindi": "hi", "Oriya": "or",
                "English": "en", "Italian": "it", "Portuguese": "pt", "Russian": "ru",
                "Japanese": "ja", "Chinese": "zh-cn", "Arabic": "ar"
            }
            dest_lang = lang_map.get(language)
            if not dest_lang:
                raise ValueError(f"Unsupported language: {language}")

            translation = self.translator.translate(text, dest=dest_lang)
            self.update_chat_display(f"Original: {text}")
            self.update_chat_display(f"Translated ({language}): {translation.text}")
            self.speak(f"Translated text: {translation.text}")
        except Exception as e:
            self.update_chat_display(f"Error translating text: {str(e)}")
            self.speak("Error translating text.")
            logging.error(f"Translation error: {str(e)}")
            raise
        finally:
            self.hide_progress_bar()

    def handle_email(self):
        dialog = Toplevel(self.root)
        dialog.title("Send Email")
        dialog.geometry("500x500")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = Frame(dialog, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        Label(frame, text="Your Email:").pack(anchor=W, pady=(0, 5))
        from_entry = tk.Entry(frame, width=50)
        from_entry.pack(fill=X, pady=(0, 10))

        Label(frame, text="Password/App Password:").pack(anchor=W, pady=(0, 5))
        password_entry = tk.Entry(frame, width=50, show="*")
        password_entry.pack(fill=X, pady=(0, 10))

        Label(frame, text="To:").pack(anchor=W, pady=(0, 5))
        to_entry = tk.Entry(frame, width=50)
        to_entry.pack(fill=X, pady=(0, 10))

        Label(frame, text="Subject:").pack(anchor=W, pady=(0, 5))
        subject_entry = tk.Entry(frame, width=50)
        subject_entry.pack(fill=X, pady=(0, 10))

        Label(frame, text="Message:").pack(anchor=W, pady=(0, 5))
        message_entry = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=50, height=10)
        message_entry.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        from_entry.focus_set()

        def submit():
            from_address = from_entry.get().strip()
            password = password_entry.get().strip()
            to_address = to_entry.get().strip()
            subject = subject_entry.get().strip()
            message = message_entry.get("1.0", tk.END).strip()
            if from_address and password and to_address and subject and message:
                dialog.destroy()
                self.send_email(from_address, password, to_address, subject, message)
            else:
                messagebox.showwarning("Invalid Input", "Please fill in all fields.")
                self.speak("Please fill in all fields.")

        Button(frame, text="Send Email", command=submit).pack(pady=10)

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def send_email(self, from_address, password, to_address, subject, message):
        """Send an email using SMTP."""
        self.update_chat_display(f"Sending email to {to_address}...")
        self.speak(f"Sending email to {to_address}")
        self.show_progress_bar()
        try:
            msg = MIMEMultipart()
            msg['From'] = from_address
            msg['To'] = to_address
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_address, password)
            server.sendmail(from_address, to_address, msg.as_string())
            server.quit()

            self.update_chat_display(f"Email sent to {to_address}")
            self.speak(f"Email sent to {to_address}")
        except Exception as e:
            self.update_chat_display(f"Error sending email: {str(e)}")
            self.speak("Error sending email.")
            logging.error(f"Email error: {str(e)}")
            raise
        finally:
            self.hide_progress_bar()

    def search_youtube_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Search YouTube")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        Label(dialog, text="Enter search query:").pack(pady=10)
        entry = tk.Entry(dialog, width=30)
        entry.pack(pady=10)
        entry.focus_set()
        def submit():
            query = entry.get().strip()
            if query:
                dialog.destroy()
                self.search_youtube(query)
        Button(dialog, text="Search", command=submit).pack(pady=10)
        entry.bind("<Return>", lambda event: submit())

    def search_google_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Search Google")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        Label(dialog, text="Enter search query:").pack(pady=10)
        entry = tk.Entry(dialog, width=30)
        entry.pack(pady=10)
        entry.focus_set()
        def submit():
            query = entry.get().strip()
            if query:
                dialog.destroy()
                self.search_google(query)
        Button(dialog, text="Search", command=submit).pack(pady=10)
        entry.bind("<Return>", lambda event: submit())

    def analyze_article_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Analyze Article")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        Label(dialog, text="Enter article URL:").pack(pady=10)
        entry = tk.Entry(dialog, width=30)
        entry.pack(pady=10)
        entry.focus_set()
        def submit():
            url = entry.get().strip()
            if url:
                dialog.destroy()
                self.analyze_article(url)
        Button(dialog, text="Analyze", command=submit).pack(pady=10)
        entry.bind("<Return>", lambda event: submit())

    def show_commands_list(self):
        commands = [
            "Open YouTube/Google/Claude/ChatGPT/Copilot/WhatsApp/VS Code",
            "Open AI Studio Live",
            "Open AI Studio New Chat",
            "Use Google AI Studio",
            "What time is it? / Current time",
            "What's the date? / Current date",
            "Weather in [city]",
            "Set reminder",
            "Set alarm",
            "Tell me a joke",
            "Good morning/evening/night",
            "Who are you?",
            "Translate [text] to [language]",
            "Send email",
            "Change wake word",
            "System status",
            "Calculate [math expression]",
            "Toggle theme",
            "Manage tasks",
            "Face Detection",
            "Object Detection",
            "Crop Disease Detection (if model available)",
            "Cloud Vision Analysis",
            "Stop Camera",
            "Capture Image",
            "Upload Image",
            "Search YouTube for [query]",
            "Search Google for [query]",
            "Analyze Article [URL]",
            "Exit / Quit / Goodbye",
            "Any other query (will search Google)"
        ]
        commands_window = Toplevel(self.root)
        commands_window.title("Available Commands")
        commands_window.geometry("400x600")
        Label(commands_window, text="Available Commands", font=("Arial", 14, "bold")).pack(pady=10)
        commands_list = scrolledtext.ScrolledText(commands_window, wrap=WORD, height=20)
        commands_list.pack(expand=True, fill=BOTH, padx=10, pady=10)
        for cmd in commands:
            commands_list.insert(tk.END, f"• {cmd}\n")
        commands_list.config(state=DISABLED)
        Button(commands_window, text="Close", command=commands_window.destroy).pack(pady=10)

    def exit_program(self):
        """Cleanly exit the program."""
        try:
            self.update_chat_display("Goodbye! Have a great day!")
            self.speak("Goodbye! Have a great day!")
            self.listening_for_wake_word = False
            self.stop_camera()
            if self.cap:
                self.cap.release()
            if self.root:
                self.root.destroy()
        except Exception as e:
            logging.error(f"Exit error: {str(e)}")

def main():
    try:
        assistant = PersonalAssistant()
        root = assistant.create_gui()
        root.mainloop()
    except Exception as e:
        logging.error(f"Main error: {str(e)}")
        messagebox.showerror("Error", f"Failed to start: {str(e)}")

if __name__ == "__main__":
    main()