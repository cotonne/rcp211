from ale_py import ALEInterface

import os
import cv2
import glob
import shutil
import re
import math
from pathlib import Path
import numpy as np
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

class Recorder:
    def __init__(self, episode: int) -> None:
        self.episode = episode
        self.dir = f'{self.episode}'
        self.current = 0
    
    def start(self) -> None:
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        os.mkdir(self.dir)
        if os.path.exists(f'{self.episode}.avi'):
            os.remove(f'{self.episode}.avi')

    def save_system(self, system):
        system.saveScreenPNG(f"{self.episode}/{self.current}.png")
        self.current += 1

    def save_RGB(self, data: torch.tensor):
        print(data.shape)
        image = Image.fromarray(data.numpy().astype(np.uint8))
        image.save(f"{self.episode}/{self.current}.png")
        self.current += 1
    
    def save_Y(self, data: torch.tensor):
        image = (255 * data.permute(1, 2, 0)).cpu().numpy().astype(np.uint8)
        cv2.imwrite(f"{self.episode}/{self.current}.png", image)
        self.current += 1

    def stop(self) -> None:
        img_array = []
        for filename in sorted(glob.glob(f'{self.episode}/*.png'), key=get_order):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
            img_array.append(img)
            img_array.append(img)
            img_array.append(img)
        print(f"Saving {len(img_array)} png as {self.episode}.avi")

        out = cv2.VideoWriter(f'{self.episode}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        shutil.rmtree(self.dir, ignore_errors=True)


class DummyRecorder:
    def __init__(self, episode: int) -> None:
        pass

    def start(self) -> None:
        pass

    def save_system(self, system):
        pass

    def save_RGB(self, data: torch.tensor):
        pass

    def save_Y(self, data: torch.tensor):
        pass

    def stop(self) -> None:
        pass