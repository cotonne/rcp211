from ale_py import ALEInterface

import os
import cv2
import glob
import shutil

class Recorder():
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

    def save(self, system: ALEInterface):
        system.saveScreenPNG(f"{self.episode}/{self.current}.png")
        self.current += 1
    
    def stop(self) -> None:
        img_array = []
        for filename in glob.glob(f'{self.episode}/*.png'):
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
