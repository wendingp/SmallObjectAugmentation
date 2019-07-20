# import glob
import math
import random
from os.path import basename, dirname, join, split

import cv2 as cv2
import numpy as np
from PIL import Image

from util import *


def find_str(filename):
    """????"""
    return dirname(filename[filename.find('train' if 'train' in filename else 'val'):])


def convert_all_boxes(shape, anno_infos, yolo_label_txt_dir):
    height, width, _ = shape
    label_file = open(yolo_label_txt_dir, 'w')
    for anno_info in anno_infos:
        target_id, x1, y1, x2, y2 = anno_info
        b = (float(x1), float(x2), float(y1), float(y2))
        bb = convert((width, height), b)
        label_file.write(str(target_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def save_crop_image(save_crop_base_dir, image_dir, idx, roi):
    crop_save_dir = join(save_crop_base_dir, find_str(image_dir))
    ensure_dir_exists(crop_save_dir)
    crop_img_save_dir = join(crop_save_dir, basename(image_dir)[:-3] + '_crop_' + str(idx) + '.jpg')
    cv2.imwrite(crop_img_save_dir, roi)


def copy_small_objects(image_dir, label_dir, save_base_dir, save_crop_base_dir,
                       save_annotation_base_dir):
    image = cv2.imread(image_dir)
    # labels = load_txt_label(label_dir)
    labels = read_label_txt(label_dir)
    if len(labels) == 0:
        return
    rescale_labels = rescale_yolo_labels(labels, image.shape)
    all_boxes = []
    # save_annotation_dir = join(save_annotation_base_dir,find_str(image_dir))
    # check_dir(save_annotation_dir)
    # save_img_dir = join(save_annotation_dir,basename(image_dir))
    # draw_annotation_to_image(image,rescale_labels,save_img_dir) #validate
    for rescale_label in rescale_labels:
        all_boxes.append(rescale_label)
        rescale_label_height, rescale_label_width = rescale_label[4] - \
            rescale_label[2], rescale_label[3] - rescale_label[1]
        if is_small_object((rescale_label_height, rescale_label_width),
                           thresh=64 * 64) and rescale_label[0] == '1':
            roi = image[rescale_label[2]:rescale_label[4], rescale_label[1]:rescale_label[3]]
            # save_crop_image(save_crop_base_dir,image_dir,idx,roi)
            new_bboxes = random_add_patches(
                rescale_label, rescale_labels, image.shape, paste_number=2, iou_thresh=0.2)
            count = 0
            for new_bbox in new_bboxes:
                count += 1
                all_boxes.append(new_bbox)
                bbox_left, bbox_top, bbox_right, bbox_bottom = new_bbox[1], new_bbox[2], new_bbox[3], new_bbox[4]
                try:
                    if count > 1:
                        roi = flip_bbox(roi)
                    # save_crop_image(save_crop_base_dir,image_dir,idx,roi_fl)
                    image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                except ValueError as e:
                    print(e)
                    continue
    dir_name = find_str(image_dir)
    save_dir = join(save_base_dir, dir_name)
    ensure_dir_exists(save_dir)
    yolo_txt_dir = join(save_dir, basename(image_dir.replace('.jpg', '_augment.txt')))
    cv2.imwrite(join(save_dir, basename(image_dir).replace('.jpg', '_augment.jpg')), image)
    convert_all_boxes(image.shape, all_boxes, yolo_txt_dir)
