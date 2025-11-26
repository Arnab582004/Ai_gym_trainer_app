

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import mediapipe as mp
import numpy as np
import threading
import speech_recognition as sr
import pyttsx3
import time
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk
CAL_PUSHUP = 0.4
CAL_SQUAT  = 0.32
CAL_CURL   = 0.2
engine = pyttsx3.init()
_engine_lock = threading.Lock()

def speak(text):
    try:
        with _engine_lock:
            engine.say(text)
            engine.runAndWait()
    except Exception:
        pass

def speak_async(text):
    threading.Thread(target=lambda: speak(text), daemon=True).start()


def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def accuracy_from_angle(angle, low, high):
    return int(np.clip(np.interp(angle, [low, high], [100, 0]), 0, 100))
class VoiceListener(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.r = sr.Recognizer()
        try:
            self.mic = sr.Microphone()
        except Exception:
            self.mic = None
        self.command = ""
        self._stop = threading.Event()

    def run(self):
        if not self.mic:
            return
        while not self._stop.is_set():
            try:
                with self.mic as source:
                    self.r.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.r.listen(source, phrase_time_limit=3)
                    said = self.r.recognize_google(audio).lower()
                    print("[VOICE]", said)
                    self.command = said
            except Exception:
                pass

    def get_command(self):
        cmd = self.command
        self.command = ""
        return cmd

    def stop(self):
        self._stop.set()
class PoseProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        # fallback without dshow
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return out, results, w, h

    def release(self):
        try:
            self.cap.release()
            self.pose.close()
        except Exception:
            pass
class GymApp:
    def __init__(self, root):
        self.root = root
        root.title("AI Gym Trainer Pro")
        self.running = False

        # Pose processor and voice
        self.processor = PoseProcessor()
        self.voice = VoiceListener()
        self.voice.start()

        # Counters and states
        self.pushup_count = 0
        self.pushup_stage = None
        self.squat_count = 0
        self.squat_stage = None
        self.curl_count = 0
        self.curl_stage = None
        self.total_calories = 0.0
        self.current_exercise = "None"
        self.workout_log = []
        self._last_tts = 0
        self.setup_ui()
        self.video_thread = None

    def setup_ui(self):
        control = ttk.Frame(self.root, padding=6)
        control.grid(row=0, column=0, sticky='ns')

        ttk.Label(control, text="Mode").grid(row=0, column=0, pady=4)
        self.mode_var = tk.StringVar(value=self.current_exercise)
        modes = ["None", "Push-ups", "Squats", "Bicep Curls", "All Exercises"]
        self.mode_menu = ttk.OptionMenu(control, self.mode_var, modes[0], *modes, command=self.set_mode)
        self.mode_menu.grid(row=1, column=0, pady=4)

        self.start_btn = ttk.Button(control, text="Start", command=self.start)
        self.start_btn.grid(row=2, column=0, pady=4, sticky='ew')
        self.stop_btn = ttk.Button(control, text="Stop", command=self.stop)
        self.stop_btn.grid(row=3, column=0, pady=4, sticky='ew')

        ttk.Separator(control).grid(row=4, column=0, pady=6, sticky='ew')

        ttk.Label(control, text="Quick Keys:").grid(row=5, column=0, pady=2)
        ttk.Label(control, text="1 Push | 2 Squat | 3 Curl | 4 All | q Quit").grid(row=6, column=0)

        ttk.Separator(control).grid(row=7, column=0, pady=6, sticky='ew')

        ttk.Button(control, text="Save Log Now", command=self.save_log).grid(row=8, column=0, pady=4, sticky='ew')
        ttk.Button(control, text="Exit", command=self.on_exit).grid(row=9, column=0, pady=4, sticky='ew')
        display = ttk.Frame(self.root, padding=6)
        display.grid(row=0, column=1)

        self.video_label = tk.Label(display)
        self.video_label.grid(row=0, column=0)

        stats = ttk.Frame(display)
        stats.grid(row=1, column=0, pady=6, sticky='ew')

        self.status_var = tk.StringVar(value=f"Mode: {self.current_exercise}")
        ttk.Label(stats, textvariable=self.status_var).grid(row=0, column=0, sticky='w')

        self.push_var = tk.StringVar(value=f"Pushups: {self.pushup_count}")
        ttk.Label(stats, textvariable=self.push_var).grid(row=1, column=0, sticky='w')

        self.squat_var = tk.StringVar(value=f"Squats: {self.squat_count}")
        ttk.Label(stats, textvariable=self.squat_var).grid(row=2, column=0, sticky='w')

        self.curl_var = tk.StringVar(value=f"Curls: {self.curl_count}")
        ttk.Label(stats, textvariable=self.curl_var).grid(row=3, column=0, sticky='w')

        self.cal_var = tk.StringVar(value=f"Calories: {self.total_calories:.1f}")
        ttk.Label(stats, textvariable=self.cal_var).grid(row=4, column=0, sticky='w')

    def set_mode(self, val):
        self.current_exercise = val
        self.status_var.set(f"Mode: {self.current_exercise}")
        tts = f"{val} mode activated" if val != "None" else "No mode"
        self.tts_debounce(tts)

    def tts_debounce(self, msg, cooldown=0.7):
        now = time.time()
        if now - self._last_tts > cooldown:
            speak_async(msg)
            self._last_tts = now

    def start(self):
        if self.running:
            return
        self.running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.tts_debounce("Starting workout")

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.tts_debounce("Stopping workout")

    def on_exit(self):
        self.stop()
        time.sleep(0.3)
        self.save_log()
        self.processor_release_and_exit()

    def save_log(self):
        if not self.workout_log:
            messagebox.showinfo("Save Log", "No workout data to save.")
            return
        df = pd.DataFrame(self.workout_log)
        fname = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files','*.csv')], initialfile=f"workout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if fname:
            df.to_csv(fname, index=False)
            messagebox.showinfo("Save Log", f"Saved to {fname}")

    def processor_release_and_exit(self):
        try:
            self.processor.release()
        except Exception:
            pass
        self.voice.stop()
        self.root.quit()

    def video_loop(self):
        self.processor = PoseProcessor()
        mp_pose_local = mp.solutions.pose

        while self.running and self.processor.cap.isOpened():
            frame = self.processor.read_frame()
            if frame is None:
                break
            cmd = self.voice.get_command()
            if cmd:
                if 'push' in cmd:
                    self.current_exercise = 'Push-ups'
                    self.mode_var.set(self.current_exercise)
                    self.tts_debounce('Pushups mode activated')
                elif 'squat' in cmd:
                    self.current_exercise = 'Squats'
                    self.mode_var.set(self.current_exercise)
                    self.tts_debounce('Squats mode activated')
                elif 'curl' in cmd:
                    self.current_exercise = 'Bicep Curls'
                    self.mode_var.set(self.current_exercise)
                    self.tts_debounce('Curls mode activated')
                elif 'all' in cmd:
                    self.current_exercise = 'All Exercises'
                    self.mode_var.set(self.current_exercise)
                    self.tts_debounce('All exercises mode activated')
                elif 'stop' in cmd or 'quit' in cmd:
                    self.running = False
                    break

            out, results, w, h = self.processor.process(frame)

            if results and results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(out, results.pose_landmarks, mp_pose_local.POSE_CONNECTIONS)
                land = results.pose_landmarks.landmark
                left_shoulder = [land[mp_pose_local.PoseLandmark.LEFT_SHOULDER].x * w, land[mp_pose_local.PoseLandmark.LEFT_SHOULDER].y * h]
                left_elbow    = [land[mp_pose_local.PoseLandmark.LEFT_ELBOW].x * w,    land[mp_pose_local.PoseLandmark.LEFT_ELBOW].y * h]
                left_wrist    = [land[mp_pose_local.PoseLandmark.LEFT_WRIST].x * w,    land[mp_pose_local.PoseLandmark.LEFT_WRIST].y * h]
                left_hip      = [land[mp_pose_local.PoseLandmark.LEFT_HIP].x * w,      land[mp_pose_local.PoseLandmark.LEFT_HIP].y * h]
                left_knee     = [land[mp_pose_local.PoseLandmark.LEFT_KNEE].x * w,     land[mp_pose_local.PoseLandmark.LEFT_KNEE].y * h]
                left_ankle    = [land[mp_pose_local.PoseLandmark.LEFT_ANKLE].x * w,    land[mp_pose_local.PoseLandmark.LEFT_ANKLE].y * h]
                right_shoulder = [land[mp_pose_local.PoseLandmark.RIGHT_SHOULDER].x * w, land[mp_pose_local.PoseLandmark.RIGHT_SHOULDER].y * h]
                right_elbow    = [land[mp_pose_local.PoseLandmark.RIGHT_ELBOW].x * w,    land[mp_pose_local.PoseLandmark.RIGHT_ELBOW].y * h]
                right_wrist    = [land[mp_pose_local.PoseLandmark.RIGHT_WRIST].x * w,    land[mp_pose_local.PoseLandmark.RIGHT_WRIST].y * h]

                elbow_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
                knee_angle_left  = calculate_angle(left_hip, left_knee, left_ankle)
                elbow_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
                if self.current_exercise in ['Push-ups', 'All Exercises']:
                    push_accuracy = accuracy_from_angle(elbow_angle_left, 40, 170)
                    if elbow_angle_left > 160:
                        self.pushup_stage = 'up'
                    if elbow_angle_left < 75 and self.pushup_stage == 'up':
                        self.pushup_stage = 'down'
                        self.pushup_count += 1
                        self.total_calories += CAL_PUSHUP
                        self.workout_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'exercise': 'pushup',
                            'rep': self.pushup_count,
                            'angle': float(elbow_angle_left),
                            'calories': CAL_PUSHUP,
                            'accuracy': push_accuracy
                        })
                        self.tts_debounce('Good pushup')
                    if push_accuracy < 60 and time.time() - self._last_tts > 0.7:
                        if elbow_angle_left > 140:
                            self.tts_debounce('Lower your chest more')
                        else:
                            self.tts_debounce('Keep your back straight')
                if self.current_exercise in ['Squats', 'All Exercises']:
                    squat_accuracy = accuracy_from_angle(knee_angle_left, 80, 165)
                    if knee_angle_left > 160:
                        self.squat_stage = 'up'
                    if knee_angle_left < 90 and self.squat_stage == 'up':
                        self.squat_stage = 'down'
                        self.squat_count += 1
                        self.total_calories += CAL_SQUAT
                        self.workout_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'exercise': 'squat',
                            'rep': self.squat_count,
                            'angle': float(knee_angle_left),
                            'calories': CAL_SQUAT,
                            'accuracy': squat_accuracy
                        })
                        self.tts_debounce('Nice squat')
                    if squat_accuracy < 60 and time.time() - self._last_tts > 0.7:
                        if knee_angle_left > 110:
                            self.tts_debounce('Go deeper in your squat')
                        else:
                            self.tts_debounce('Keep your chest up')
                if self.current_exercise in ['Bicep Curls', 'All Exercises']:
                    curl_accuracy = accuracy_from_angle(elbow_angle_right, 40, 150)
                    if elbow_angle_right > 150:
                        self.curl_stage = 'down'
                    if elbow_angle_right < 60 and self.curl_stage == 'down':
                        self.curl_stage = 'up'
                        self.curl_count += 1
                        self.total_calories += CAL_CURL
                        self.workout_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'exercise': 'curl',
                            'rep': self.curl_count,
                            'angle': float(elbow_angle_right),
                            'calories': CAL_CURL,
                            'accuracy': curl_accuracy
                        })
                        self.tts_debounce('Nice curl')
                    if curl_accuracy < 60 and time.time() - self._last_tts > 0.7:
                        self.tts_debounce('Curl slowly and fully contract')
            self.status_var.set(f"Mode: {self.current_exercise}")
            self.push_var.set(f"Pushups: {self.pushup_count}")
            self.squat_var.set(f"Squats: {self.squat_count}")
            self.curl_var.set(f"Curls: {self.curl_count}")
            self.cal_var.set(f"Calories: {self.total_calories:.1f}")

            img = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            # Avoid garbage collection
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        time.sleep(0.01)
        try:
            self.root.update_idletasks()
        except tk.TclError:
            return
        self.processor.release()
        self.running = False
        if self.workout_log:
            fname = f"workout_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            pd.DataFrame(self.workout_log).to_csv(fname, index=False)
            print(f"Saved workout log to {fname}")
if __name__ == '__main__':
    root = tk.Tk()
    app = GymApp(root)
    def on_key(e):
        k = e.char
        if k == '1':
            app.current_exercise = 'Push-ups'; app.mode_var.set(app.current_exercise); app.tts_debounce('Pushups mode activated')
        elif k == '2':
            app.current_exercise = 'Squats'; app.mode_var.set(app.current_exercise); app.tts_debounce('Squats mode activated')
        elif k == '3':
            app.current_exercise = 'Bicep Curls'; app.mode_var.set(app.current_exercise); app.tts_debounce('Curls mode activated')
        elif k == '4':
            app.current_exercise = 'All Exercises'; app.mode_var.set(app.current_exercise); app.tts_debounce('All exercises mode activated')
        elif k == 'q':
            app.on_exit()

    root.bind('<Key>', on_key)

    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()

