import os
import cv2
import time
import math
import numpy as np
from pathlib import Path
from enum import IntFlag
from typing import List
from ultralytics import YOLO
from ultralytics.engine.results import Results

#TODO: ENV
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = Path(APP_DIR, 'frames')
WEIGHTS_PATH = Path(APP_DIR, 'best.pt')

#TODO: work interval
WORK_INTERVAL_SECONDS = 5
SAVE_INTERVAL_SECONDS = 30
SAVE_INTERVAL_IF_NOT_DETECTED_SECONDS = 1
FAILED_FRAMES = 10
AVERAGE_QUANTITY_FPS = 10
RATE = 0.85

IMGSZ = 640

MODE = 0b10111

class Modes(IntFlag):
    DETECT = 1                          # 0b00001
    FRAME_STAT = 2                      # 0b00010
    FRAME_SHOW = 4                      # 0b00100
    FRAME_SAVE = 8                      # 0b01000
    FRAME_SAVE_IF_NOT_DETECTED = 16     # 0b10000


#TODO: not work
class FpsModel:
    def __init__(self):
        self.fps_start_time = 0
        self.frame_count = 0
        self.fps_vals = []

    def start(self):
        self.fps_start_time = time.time()

    def next(self):
        fps = self.frame_count / self.fps_start_time
        self.fps_vals.append(fps)

        if len(self.fps_vals) == AVERAGE_QUANTITY_FPS:
            self.fps_vals.pop(0)

        self.fps_start_time = time.time()
        self.frame_count = 0

    def reset(self):
        self.fps_start_time = time.time()
        self.frame_count = 0
        self.fps_vals = []

    def get(self):
        return int(sum(self.fps_vals) / len(self.fps_vals))

class FrameSaver:
    def __init__(self, frames_dir):
        self.frame_dir = frames_dir
        self.switch = 0
        self.time_count = 0
        self.last_time = int(time.time())

    def smart_save(self, frame: np.ndarray) -> None:
        diff = int(time.time()) - self.last_time
        if diff > self.time_count:
            self.time_count += math.ceil(diff / 2)
            self.last_time = int(time.time())

    def save(self, frame: np.ndarray) -> None:
        if not os.path.isdir(self.frame_dir):
            os.makedirs(self.frame_dir)
        filename = f'{time.time()}.jpg'
        frame_path = Path(self.frame_dir, filename)
        cv2.imwrite(frame_path, frame)

    def reset(self):
        self.time_count = 0
        self.last_time = int(time.time())

def put_detect_stat(frame: np.ndarray, detect_res: List[Results]) -> None:
    for r in detect_res:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = b.conf[0]
            cls = int(b.cls[0])
            label = f"{r.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

def put_fps_stat(frame: np.ndarray, fps: int) -> None:
    cv2.putText(
        frame, f'FPS: {fps}', (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

def start(frames_dir: Path, mode: int) -> None:
    mode = Modes(mode)
    predict_res = None
    saver = FrameSaver(FRAMES_DIR)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception('Error: failed to open camera')

    model = None
    if Modes.DETECT in mode:
        if not WEIGHTS_PATH.exists():
            raise Exception(f'Error: model weights not found at {WEIGHTS_PATH}')
        model = YOLO(str(WEIGHTS_PATH))

    try:
        while True:
            ret, frame = cap.read()

            if Modes.DETECT in mode:
                _frame = cv2.resize(frame, (IMGSZ, IMGSZ))
                predict_res = model.predict(_frame, conf=RATE, verbose=False)

            if Modes.FRAME_SAVE_IF_NOT_DETECTED in mode:
                if predict_res:
                    if not len([box for result in predict_res for box in result.boxes]):
                        saver.smart_save(frame)
                    else:
                        saver.reset()

            if Modes.FRAME_STAT in mode:
                pass

            if Modes.FRAME_SHOW in mode:
                cv2.imshow('frame', frame)

            if Modes.FRAME_SAVE in mode:
                saver.save(frame)

            if Modes.DETECT in mode:
                pass

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break
            elif key & 0xff == ord('s'):
                saver.save(frame)

    except Exception as e:
        print(e)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    start(frames_dir=FRAMES_DIR, mode=MODE)