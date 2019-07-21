import os
import random
from os.path import basename, dirname, join

import cv2
import numpy as np


def read_label_txt(label_path):
    labels = []
    with open(label_path) as fp:
        for line in fp.readlines():
            labels.append(line.strip().split(' '))
    return labels


def flip_bbox(roi):
    return roi[:, ::-1, :]


# def save_crop_image(save_crop_base_dir, image_dir, idx, roi):
#     crop_save_dir = join(save_crop_base_dir, find_str(image_dir))
#     ensure_dir_exists(crop_save_dir)
#     crop_img_save_dir = join(crop_save_dir, basename(image_dir)[:-3] + '_crop_' + str(idx) + '.jpg')
#     cv2.imwrite(crop_img_save_dir, roi)


def find_str(filename):
    """????"""
    return dirname(filename[filename.find('train' if 'train' in filename else 'val'):])


def convert_all_boxes(shape, anno_infos, yolo_label_txt_dir):
    height, width, _ = shape
    label_file = open(yolo_label_txt_dir, 'w')
    for target_id, x1, y1, x2, y2 in anno_infos:
        b = (float(x1), float(x2), float(y1), float(y2))
        bb = convert((width, height), b)
        label_file.write(str(target_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def is_small_object(bbox, thresh):
    """check if the given bbox is small object

    Arguments:
        bbox {2d tuple} -- h, w of bbox
        thresh {float} -- given threshold

    Returns:
        bool -- if is small object
    """
    return bbox[0] * bbox[1] <= thresh


def load_txt_label(label_txt_path):
    return np.loadtxt(label_txt_path, dtype=str)


def load_txt_labels(label_dir):
    return [load_txt_label(label) for label in label_dir]


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Makes new dir: ", dir)


def rescale_yolo_labels(labels, img_shape):
    height, width, nchannel = img_shape
    rescale_boxes = []
    for box in list(labels):
        x_c = float(box[1]) * width
        y_c = float(box[2]) * height
        w = float(box[3]) * width
        h = float(box[4]) * height
        x_left = x_c - w * .5
        y_left = y_c - h * .5
        x_right = x_c + w * .5
        y_right = y_c + h * .5
        rescale_boxes.append([box[0], int(x_left), int(y_left), int(x_right), int(y_right)])
    return rescale_boxes


def draw_annotation_to_image(img, annotation, save_img_dir):
    for anno in annotation:
        cl, x1, y1, x2, y2 = anno
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, cl, (int((x1 + x2) / 2), y1 - 5), font, fontScale=0.8, color=(0, 0, 255))
    cv2.imwrite(save_img_dir, img)


def bbox_iou(box1, box2):
    cl, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    cl, b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width <= 0 and inter_height <= 0:  # strong condition
        return 0
    inter_area = inter_width * inter_height
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def uniform_sample(search_space):
    """Uniformly sample bboxes

    Arguments:
        search_space (4 num) -- range of search

    Returns:
        center of new boxes
    """
    search_x_left, search_y_left, search_x_right, search_y_right = search_space
    new_bbox_x_center = random.randint(search_x_left, search_x_right)
    new_bbox_y_center = random.randint(search_y_left, search_y_right)
    return [new_bbox_x_center, new_bbox_y_center]


def sampling_new_bbox_center_point(img_shape, bbox):
    # sampling space
    height, width, n_channels = img_shape
    cl, x_left, y_left, x_right, y_right = bbox
    # bbox_w, bbox_h = x_right - x_left, y_right - y_left
    # left top
    if x_left <= width / 2:
        search_x_left, search_y_left, search_x_right, search_y_right = width * 0.6, height / 2, width * 0.75, height * 0.75
    else:
        search_x_left, search_y_left, search_x_right, search_y_right = width * 0.25, height / 2, width * 0.5, height * 0.75
    return [search_x_left, search_y_left, search_x_right, search_y_right]


def img_paths2label_paths(img_paths):
    """get labels' path from images' path"""
    return [img_path.replace('.jpg', '.txt') for img_path in img_paths]


def random_add_patches(bbox, rescale_boxes, shape, paste_number, iou_thresh):
    # temp = []
    # for rescale_bbox in rescale_boxes:
    #     temp.append(rescale_bbox)
    cl, x_left, y_left, x_right, y_right = bbox
    bbox_w, bbox_h = x_right - x_left, y_right - y_left
    center_search_space = sampling_new_bbox_center_point(shape, bbox)
    n_success = 0
    new_bboxes = []
    while n_success < paste_number:
        new_bbox_x_center, new_bbox_y_center = uniform_sample(center_search_space)
        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = int(
            new_bbox_x_center - 0.5 * bbox_w), int(
            new_bbox_y_center - 0.5 * bbox_h), int(new_bbox_x_center + 0.5 * bbox_w), int(
            new_bbox_y_center + 0.5 * bbox_h)
        new_bbox = [cl, new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right]
        ious = [bbox_iou(new_bbox, bbox_t) for bbox_t in rescale_boxes]
        if max(ious) > iou_thresh:
            continue
        n_success += 1
        # temp.append(new_bbox)
        new_bboxes.append(new_bbox)

    return new_bboxes


def copy_small_objects(img_path, label_path, save_base_dir, save_crop_base_dir,
                       save_annotation_base_dir):
    img = cv2.imread(img_path)
    labels = read_label_txt(label_path)
    assert labels, "No labels read!"

    rescale_labels = rescale_yolo_labels(labels, img.shape)
    all_boxes = []
    # save_annotation_dir = join(save_annotation_base_dir,find_str(img_path))
    # check_dir(save_annotation_dir)
    # save_img_dir = join(save_annotation_dir,basename(img_path))
    # draw_annotation_to_image(img, rescale_labels, save_img_dir)  # validate
    for rescale_label in rescale_labels:
        all_boxes.append(rescale_label)
        rescale_label_height, rescale_label_width = rescale_label[4] - \
            rescale_label[2], rescale_label[3] - rescale_label[1]
        if is_small_object((rescale_label_height, rescale_label_width),
                           thresh=64 * 64) and rescale_label[0] == '1':
            roi = img[rescale_label[2]:rescale_label[4], rescale_label[1]:rescale_label[3]]
            # save_crop_image(save_crop_base_dir, img_path, idx, roi)
            new_bboxes = random_add_patches(
                rescale_label, rescale_labels, img.shape, paste_number=2, iou_thresh=0.2)
            count = 0
            for new_bbox in new_bboxes:
                count += 1
                all_boxes.append(new_bbox)
                bbox_left, bbox_top, bbox_right, bbox_bottom = new_bbox[1], new_bbox[2], new_bbox[3], new_bbox[4]
                try:
                    if count > 1:
                        roi = flip_bbox(roi)
                    # save_crop_image(save_crop_base_dir, img_path, idx, roi_fl)
                    img[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                except ValueError as e:
                    print(e)

    dir_name = find_str(img_path)
    save_dir = join(save_base_dir, dir_name)
    ensure_dir_exists(save_dir)
    yolo_txt_dir = join(save_dir, basename(img_path.replace('.jpg', '_augment.txt')))
    cv2.imwrite(join(save_dir, basename(img_path).replace('.jpg', '_augment.jpg')), img)
    convert_all_boxes(img.shape, all_boxes, yolo_txt_dir)
