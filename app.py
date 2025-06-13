#!/usr/bin/env python
# -- coding: utf-8 --
import csv
import copy
import argparse
import itertools
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import smtplib
from email.message import EmailMessage

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import google.protobuf.message_factory as mf
if not hasattr(mf, 'GetMessageClass'):
    mf.GetMessageClass = mf.GetPrototype


# =================== Email Settings (from reference code) =================== #
SENDER_EMAIL = "tensortitans2612@gmail.com"
SENDER_PASSWORD = "hjcy lblh gwhv jmzk"  # Use app-specific passwords in production
DEFAULT_RECEIVERS = [
    "siddhantpatil1543@gmail.com",
    "siddhantpatil1540@gmail.com"
]

def send_emergency_email(suspicious_frame=None):
    """
    Send an emergency email alert with an optional image attachment (the suspicious frame).
    """
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(DEFAULT_RECEIVERS)
    msg["Subject"] = "Emergency Alert !!! Threat Hand Signs detected."
    msg.set_content("Emergency Alert !!! Threat Hand Signs detected.")

    # Attach the image if provided
    if suspicious_frame is not None:
        ret, buffer = cv.imencode('.jpg', suspicious_frame)
        if ret:
            img_data = buffer.tobytes()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename="threat.jpg")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Emergency Email sent successfully to:", ", ".join(DEFAULT_RECEIVERS))
    except Exception as e:
        print("❌ Failed to send email:", e)

# ---------------------- Original Functions ----------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)
    args = parser.parse_args()
    return args

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)
        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

# ---------------------- End of Original Functions ----------------------

# ---------------------- Flask Application ----------------------
from flask import Flask, Response, render_template_string

app = Flask(__name__)

def generate_frames():
    # Use command-line arguments for settings (can be overridden)
    global alert_triggered
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # FPS Measurement
    cvFpsCalc_obj = CvFpsCalc(buffer_len=10)

    # Coordinate and gesture history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0
    number = 0  # Default since keyboard input is not available

    email_sent = False  # Flag to ensure the email is sent only once
    help_count = 0      # Counter for the number of "help" detections
    alert_triggered = True

    while True:
        fps = cvFpsCalc_obj.get()
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                gesture_name = keypoint_classifier_labels[hand_sign_id]
                print(f"Detected Gesture: {gesture_name}")

                # Increment counter if "Help" gesture detected (case-insensitive)
                if gesture_name.strip().lower() == "help":
                    help_count += 1
                    print(f"Help gesture count: {help_count}")
                    # When the help gesture is detected for the 10th time, send email only once.
                    if help_count == 20 and not email_sent:
                        send_emergency_email(debug_image)
                        email_sent = True
                        alert_triggered = True

                # For the "point" gesture (example checking against a string "x")
                if str(hand_sign_id) == "x":  
                    if len(landmark_list) > 8:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                fg_text = (point_history_classifier_labels[most_common_fg_id[0][0]]
                           if most_common_fg_id else "")
                debug_image = draw_info_text(debug_image, brect, handedness, 
                                             keypoint_classifier_labels[hand_sign_id], fg_text)
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Encode frame as JPEG
        ret2, buffer = cv.imencode('.jpg', debug_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv.destroyAllWindows()

@app.route('/')
def index():
    # Improved HTML with sidebar and dark theme.
    return render_template_string('''
   <!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hand Gesture Recognition</title>
    <style>
      /* Global Styles */
      body {
        background: url("static/apple.jpg") no-repeat center center;
        background-size: cover;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
      }
      /* Top Navbar */
      .topnav {
        margin-left: 250px;
        background: linear-gradient(135deg, #1f1f1f, #0d0d0d);
        padding: 10px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #444444;
        box-shadow: 0 2px 8px rgba(0,0,0,0.6);
      }
      .topnav img {
        height: 60px;
      }
      /* Sidebar Styling */
      .sidenav {
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        width: 250px;
        background: linear-gradient(135deg, #2a2a2a, #1f1f1f);
        padding: 30px 20px;
        overflow-y: auto;
        box-shadow: 2px 0 8px rgba(0,0,0,0.8);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }
      .sidenav:hover {
        transform: translateX(5px);
        box-shadow: 4px 0 16px rgba(0,0,0,0.9);
      }
      .sidenav h3 {
        margin: 0;
        padding-bottom: 20px;
        color: #ffffff;
        text-align: center;
      }
      .sidenav .nav-link {
        display: block;
        padding: 12px 10px;
        margin-bottom: 12px;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 8px;
        text-align: left;
        font-weight: bold;
        background: rgba(255,255,255,0.05);
        text-decoration: none;
        transition: background 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease, color 0.3s ease;
      }
      .sidenav .nav-link:hover {
        background: #03dac6;
        color: #000000;
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.7);
      }
      /* Content Area */
      .content {
        margin-left: 270px;
        padding: 30px;
        text-align: center;
      }
      /* Video Container */
      .video-container {
        margin-top: 20px;
        border: 2px solid #03dac6;
        border-radius: 15px;
        overflow: hidden;
        display: inline-block;
        box-shadow: 0 0 10px rgba(3, 218, 198, 0.5);
      }
      .video-container img {
        width: 80vw;
        max-width: 960px;
        height: auto;
      }
      /* Button Styles */
      button {
        background: transparent;
        border: none;
        color: #f0f0f0;
        cursor: pointer;
        padding: 30px 80px;
        font-size: 16px;
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 50px auto 20px;
        white-space: nowrap;
        transition: all 0.3s ease;
      }
      button::after,
      button::before {
        content: '';
        position: absolute;
        bottom: 0;
        right: 0;
        z-index: -1;
        transition: all 0.4s;
      }
      button::before {
        transform: translate(0, 0);
        width: 100%;
        height: 100%;
        background: #28282d;
        border-radius: 10px;
      }
      button::after {
        transform: translate(10px, 10px);
        width: 35px;
        height: 35px;
        background: #ffffff15;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-radius: 50px;
      }
      button:hover::before {
        transform: translate(5%, 20%);
        width: 110%;
        height: 110%;
      }
      button:hover::after {
        transform: translate(0, 0);
        width: 100%;
        height: 100%;
        border-radius: 10px;
      }
      button:active::after {
        transform: translate(0, 5%);
        transition: 0s;
      }
      /* Heading Styles */
      h1 {
        font-size: 28px;
        margin-top: 90px;
        transition: color 0.3s ease-in-out, transform 0.3s ease-in-out;
      }
      h1:hover {
        color: #03dac6;
        transform: scale(1.05);
      }
    </style>
  </head>
  <body>
    <!-- Top Navbar -->
    <nav class="topnav">
      <div class="logo">
        <img src="static/logo-ct.png" alt="Logo">
      </div>
    </nav>
    <!-- Sidebar -->
    <aside class="sidenav">
      <h3>Menu</h3>
      <a class="nav-link" href="#">Home</a>
       <a class="nav-link" href="#">Missing Person Detection</a>
       <a class="nav-link" href="#">Logout</a>
    </aside>
    <!-- Content -->
    <div class="content">
      <h1>Hand Gesture Recognition</h1>
      <button id="start-btn">Start Video Feed</button>
      <div class="video-container" id="video-container" style="display: none;">
        <img id="video-feed" src="" alt="Video Feed">
      </div>
    </div>
    <script>
      document.getElementById("start-btn").addEventListener("click", function() {
        document.getElementById("video-feed").src = "/video_feed";
        document.getElementById("video-container").style.display = "block";
      });
    </script>
  </body>
</html>
''')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app.
    app.run(host='0.0.0.0', debug=True)
