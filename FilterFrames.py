import os
from pathlib import Path

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = Path(APP_DIR, 'frames')
PER_FILES = 2
ITERATIONS = 1


def delete_file():
    itr = 0
    while itr < ITERATIONS:
        for idx, file in enumerate(FRAMES_DIR.glob('*.jpg')):
            if idx % PER_FILES != 0:
                file.unlink()
        itr += 1


if __name__ == '__main__':
    # delete_file()

    print(f'{FRAMES_DIR}: {len(list(FRAMES_DIR.glob('*.jpg')))} frames')