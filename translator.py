import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading
import queue
from collections import deque

# 初始化TTS引擎，并设置语速属性，降低语速
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 50)

# 加载训练好的YOLOv8模型
model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

last_output_time = 0
output_interval = 2  # 每2秒进行一次语音输出
detected_letter = None

colors = [
    (255, 182, 193), (135, 206, 250), (144, 238, 144), (255, 228, 181), (221, 160, 221), (175, 238, 238),
    (240, 128, 128), (152, 251, 152), (173, 216, 230), (255, 239, 213), (255, 160, 122), (224, 255, 255),
    (250, 128, 114), (124, 252, 0), (135, 206, 235), (255, 245, 238), (221, 160, 221), (175, 238, 238),
    (240, 230, 140), (255, 222, 173), (255, 192, 203), (240, 255, 240), (255, 228, 225), (245, 245, 220),
    (245, 255, 250), (255, 228, 196)]

# 创建一个队列来存储检测到的字母
letter_queue = queue.Queue()

# 创建一个锁来防止多线程同时调用engine.runAndWait()
lock = threading.Lock()

# 语音输出线程函数
def speak_letters():
    while True:
        letter = letter_queue.get()
        if letter is None:
            break
        with lock:
            engine.say(letter)
            engine.runAndWait()
        # 添加适当的延迟确保语音输出不会卡壳
        time.sleep(0.5)

# 创建并启动语音输出线程
threading.Thread(target=speak_letters, daemon=True).start()

window_size = 5
detected_letters = deque(maxlen=window_size)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # 使用YOLO模型进行预测
    results = model(frame)

    new_detected_letter = None

    # 预测结果
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy()  
        class_ids = result.boxes.cls.cpu().numpy()  

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf < 0.5:  
                continue

            x1, y1, x2, y2 = map(int, box)
            new_detected_letter = model.names[int(class_id)]
            label = f"{new_detected_letter}: {conf:.2f}"
            color = colors[int(class_id) % len(colors)]  

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_ymin = max(y1, label_size[1] + 10)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, cv2.FILLED)
            
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            print(f"Detected: {label}")

    current_time = time.time()

    if new_detected_letter:
        detected_letters.append(new_detected_letter)

    if len(detected_letters) == window_size and all(letter == detected_letters[0] for letter in detected_letters):
        stable_letter = detected_letters[0]
        if stable_letter != detected_letter or current_time - last_output_time >= output_interval:
            last_output_time = current_time
            detected_letter = stable_letter
            print(f"Speaking: {detected_letter}")  
            letter_queue.put(detected_letter)
        detected_letters.clear()

    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # 按下'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
letter_queue.put(None)
