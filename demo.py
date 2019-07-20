import os
from os.path import join

from tqdm import tqdm

from aug import copy_small_objects
from Helpers import replace_labels
from util import *


def main():
    base_dir = os.getcwd()
    save_base_dir = join(base_dir, 'save')
    save_crop_base_dir = join(base_dir, 'save_crop')
    save_annotation_base_dir = join(base_dir, 'save_annotation')
    ensure_dir_exists(save_base_dir)
    imgs_dir = [f.strip() for f in open(join(base_dir, 'train.txt')).readlines()]
    labels_dir = replace_labels(imgs_dir)
    for image_dir, label_dir in tqdm(zip(imgs_dir, labels_dir)):
        copy_small_objects(image_dir, label_dir, save_base_dir,
                           save_crop_base_dir, save_annotation_base_dir)


if __name__ == "__main__":
    main()
