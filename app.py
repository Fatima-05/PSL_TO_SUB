import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import collections
import time
import urllib.request
import os
from PIL import Image, ImageDraw, ImageFont

model = tf.keras.models.load_model("psl_alphabet_model.h5")

#class names list in case .txt file is missing 
class_names_default =[
    'ء', 'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ',
    'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ',
    'ف', 'ق', 'ل', 'م', 'ن', 'و', 'ٹ', 'پ', 'چ', 'ڈ',
    'ڑ', 'ژ', 'ک', 'گ', 'ہ', 'ی', 'ے'
]

if os.path.exists("class_names.txt"):
    with open("class_names.txt", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"loaded {len(class_names)} classes from class_names.txt :D")
else:
    class_names = class_names_default
    print("class_names.txt not found :( so using hardcoded list")

print(f"{len(class_names)} classes ready :D")

#must match normalise_sample() used during training
def landmarks_to_training_space(hand_landmarks, frame_w, frame_h) -> np.ndarray:
    coords = np.array(
        [[lm.x * frame_w, lm.y * frame_h] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )
    coords -= coords[0]
    spread = np.max(np.abs(coords))
    if spread > 0:
        coords = coords / spread * 150.0
    coords += 150.0
    coords = np.clip(coords, 0.0, 300.0)  
    return coords.flatten().astype(np.float32)

#download font path if it's nto present
NOTO_PATH = "NotoNaskhArabic-Regular.ttf"
if not os.path.exists(NOTO_PATH):
    try:
        urllib.request.urlretrieve(
            "https://github.com/google/fonts/raw/main/ofl/notonaskharabic/NotoNaskhArabic-Regular.ttf",
            NOTO_PATH
        )
        print("Font downloaded")
    except Exception as e:
        print(f"download failed: {e} :(")
        print("manually place NotoNaskhArabic-Regular.ttf in folder :(")

FONT_PATHS =[
    NOTO_PATH,
    "C:/Windows/Fonts/Nirmala.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

def load_font(size):
    for path in FONT_PATHS:
        try:
            f = ImageFont.truetype(path, size)
            print(f"font: {path} @ {size}px")
            return f
        except Exception:
            continue
    print(f"no font found at {size}px so using PIL default :(")
    return ImageFont.load_default()

font_large = load_font(80)
font_medium = load_font(44)

#using PIL to read urdu text
def put_urdu(frame_bgr, text, xy, font, color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

CONF_THRESHOLD = 40.0
SMOOTHING_WINDOW = 10
SUBTITLE_HOLD_S = 2.0

sign_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
subtitle_text = ""
subtitle_ts = 0.0


def majority_vote(buf):
    if not buf:
        return None, 0.0
    signs = [s for s, _ in buf]
    best = max(set(signs), key=signs.count)
    confs = [c for s, c in buf if s == best]
    return best, float(np.mean(confs))


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

if not cap.isOpened():
    print("can't open webcam")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    sign, conf = None, 0.0

    if result.multi_hand_landmarks:
        hand_lms = result.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame, hand_lms, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        kp = landmarks_to_training_space(hand_lms, w, h)
        pred = model.predict(kp.reshape(1, 42), verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx]) * 100.0
        sign = class_names[idx]

        top3 = np.argsort(pred)[::-1][:3]

        if conf >= CONF_THRESHOLD:
            sign_buffer.append((sign, conf))

    
    smooth_sign, smooth_conf = majority_vote(sign_buffer)
    if smooth_sign and smooth_conf >= CONF_THRESHOLD:
        subtitle_text = smooth_sign
        subtitle_ts = time.time()

    if sign:
        color_bgr = (0, 200, 0) if conf >= CONF_THRESHOLD else (0, 140, 255)
        color_rgb = (0, 200, 0) if conf >= CONF_THRESHOLD else (255, 140, 0)
        cv2.putText(frame, "Sign:", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{conf:.1f}%", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2, cv2.LINE_AA)
        frame = put_urdu(frame, sign, (130, 10), font_medium, color=color_rgb)
    else:
        cv2.putText(frame, "no hand detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 220), 2, cv2.LINE_AA)

    #sub bar
    if subtitle_text and (time.time() - subtitle_ts) < SUBTITLE_HOLD_S:
        bar_h = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        try:
            bbox = font_large.getbbox(subtitle_text)
            char_w = bbox[2] - bbox[0]
        except Exception:
            char_w = 60
        x_pos = max(10, w // 2 - char_w // 2)
        frame = put_urdu(frame, subtitle_text,
                         (x_pos, h - bar_h + 15),
                         font_large, color=(255, 255, 255))

        cv2.putText(frame, f"{smooth_conf:.1f}%", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)

    cv2.putText(frame, "Q: quit", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.namedWindow("PSL to Subtitles", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("PSL to Subtitles",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("PSL to Subtitles", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
