import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
# import random
import math
from tqdm import tqdm


def load_images(path):
    imgs = []
    images = glob.glob(path)
    for index in range(len(images)):
        image = cv2.cvtColor(cv2.imread(images[index]), cv2.COLOR_BGR2RGB)
        imgs.append(image)
        # imgs.append(cv2.resize(image,(1280,720)))
    return imgs


def read_images(path):
    images = glob.glob(path)
    return images


def load_images_from_path(path):
    imgs = []
    for p in tqdm(path):
        image = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        imgs.append(image)
    return imgs


def replace_labels(path):
    label_path = []
    for p in path:
        label_path.append(p.replace('.jpg', '.txt'))
    return label_path
