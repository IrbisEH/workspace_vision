import os
import shutil
from pathlib import Path
from ultralytics import YOLO


TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = Path(TRAIN_DIR, 'data.yaml')
MODEL = 'yolo11l.pt'
EPOCHS = 350
BATCH = 8
IMGSZ = 640
DEVICE = 0

EMPTY_FOLDER = Path(TRAIN_DIR, 'empty')
TRAIN_FOLDER = Path(TRAIN_DIR, 'train')
VAL_FOLDER = Path(TRAIN_DIR, 'valid')

EMPTY_TRAIN = 0.6
EMPTY_VAL = 0.4

IS_ADD_EMPTY = True

def process(src_img_file, target_img_file, label_file):
    shutil.copy(src_img_file, target_img_file)
    with(open(label_file, 'w')):
        pass

def add_empty():
    sum_empty_num = len(list(EMPTY_FOLDER.glob('*.jpg')))
    train_empty_num = int(sum_empty_num * EMPTY_TRAIN)
    val_empty_num = int(sum_empty_num * EMPTY_VAL)

    count = 0
    for img_file in EMPTY_FOLDER.glob('*.jpg'):
        if count >= sum_empty_num:
            break

        img_file_name = img_file.name
        lbl_file_name = f'{img_file.stem}.txt'

        if train_empty_num > 0:
            train_img = Path(TRAIN_FOLDER, 'images', img_file_name)
            train_lbl = Path(TRAIN_FOLDER, 'labels', lbl_file_name)
            process(img_file, train_img, train_lbl)
            train_empty_num -= 1


        if val_empty_num > 0:
            valid_img = Path(VAL_FOLDER, 'images', img_file_name)
            valid_lbl = Path(VAL_FOLDER, 'labels', lbl_file_name)
            process(img_file, valid_img, valid_lbl)
            val_empty_num -= 1


if __name__ == '__main__':
    if IS_ADD_EMPTY:
        add_empty()

    model = YOLO(MODEL)
    model.train(data=YAML_PATH, epochs=EPOCHS, imgsz=IMGSZ, device=DEVICE, batch=BATCH)
