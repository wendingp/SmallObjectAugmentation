import os
from os.path import join

from tqdm import tqdm

from aug import copy_small_objects, ensure_dir_exists, img_paths2label_paths


def main():
    base_dir = os.getcwd()
    save_base_dir = join(base_dir, 'save')
    save_crop_base_dir = join(base_dir, 'save_crop')
    save_annotation_base_dir = join(base_dir, 'save_annotation')
    ensure_dir_exists(save_base_dir)
    img_paths = [f.strip() for f in open(join(base_dir, 'train.txt')).readlines()]
    label_paths = img_paths2label_paths(img_paths)
    for img_path, label_path in tqdm(zip(img_paths, label_paths)):
        copy_small_objects(img_path, label_path, save_base_dir,
                           save_crop_base_dir, save_annotation_base_dir)


if __name__ == "__main__":
    main()
