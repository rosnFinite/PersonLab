import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi
from groundtruth import map_coco_to_personlab

flip_prob = 0.5
max_degree = 30.
max_scale = 2.
min_scale = 0.8
max_offset = 20.0
target_dist = 0.8
scale_prob = 1.
IMAGE_SHAPE = (403, 403, 3)
RIGHT_KP = [1, 2, 3,  7,  8,  9, 13, 15]
LEFT_KP = [4, 5, 6, 10, 11, 12, 14, 16]


class AugmentationSelection:
    def __init__(self, flip=False, degree=0., crop=(0, 0), scale=1.):
        self.flip = flip
        self.degree = degree
        self.crop = crop
        self.scale = scale

    @staticmethod
    def random():
        flip = np.random.uniform(0., 1.) > flip_prob   # Flip with probability of 50%
        degree = np.random.uniform(-1., 1.) * max_degree  # Rotate between -30° and 30°
        scale = (max_scale - min_scale) * np.random.uniform(0., 1.) + min_scale if np.random.uniform(0., 1.) < \
            scale_prob else 1.# Scale factor between 0.8 and 2.0
        x_offset = int(np.random.uniform(-1., 1.) * max_offset)  # Offset between -20px and 20px
        y_offset = int(np.random.uniform(-1., 1.) * max_offset)  # Offset between -20px and 20px

        return AugmentationSelection(flip, degree, (x_offset, y_offset), scale)

    @staticmethod
    def unrandom():
        flip = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0
        return AugmentationSelection(flip, degree, (x_offset, y_offset), scale)

    def affine(self, center=(IMAGE_SHAPE[1]//2, IMAGE_SHAPE[0]//2)):
        A = self.scale * cos(self.degree / 180. * pi)
        B = self.scale * sin(self.degree / 180. * pi)

        scale_size = target_dist / self.scale

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array([[1., 0., -center_x],
                                [0., 1., -center_y],
                                [0., 0., 1.]])

        rotate = np.array([[A, B, 0],
                           [-B, A, 0],
                           [0, 0, 1.]])

        scale = np.array([[scale_size, 0, 0],
                          [0, scale_size, 0],
                          [0, 0, 1.]])

        flip = np.array([[-1 if self.flip else 1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

        center2center = np.array([[1., 0., IMAGE_SHAPE[1] // 2],
                                  [0., 1., IMAGE_SHAPE[0] // 2],
                                  [0., 0., 1.]])

        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)
        return combined[0:2]


class Transformer:
    @staticmethod
    def transform(img, masks, keypoints, aug=AugmentationSelection.random()):

        # warp picture and mask
        M = aug.affine(center=(img.shape[1] // 2, img.shape[0] // 2))
        cv_shape = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

        img = cv2.warpAffine(img, M, cv_shape, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(127, 127, 127))
        out_masks = np.zeros(cv_shape[::-1] + (masks.shape[-1],))
        for i in range(masks.shape[-1]):
            out_masks[:, :, i] = cv2.warpAffine(masks[:, :, i], M, cv_shape, flags=cv2.INTER_CUBIC,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        masks = out_masks

        # warp key points
        keypoints = map_coco_to_personlab(keypoints)
        original_points = keypoints.copy()
        # print keypoints
        original_points[:, :, 2] = 1  # we reuse 3rd column in completely different way here, it is hack
        converted_points = np.matmul(M, original_points.transpose([0, 2, 1])).transpose([0, 2, 1])
        keypoints[:, :, 0:2] = converted_points

        cropped_kp = keypoints[:, :, 0] >= IMAGE_SHAPE[1]
        cropped_kp = np.logical_or(cropped_kp, keypoints[:, :, 1] >= IMAGE_SHAPE[0])
        cropped_kp = np.logical_or(cropped_kp, keypoints[:, :, 0] < 0)
        cropped_kp = np.logical_or(cropped_kp, keypoints[:, :, 1] < 0)

        keypoints[cropped_kp, 2] = 0

        # we just made image flip, i.e. right leg just became left leg, and vice versa
        if aug.flip:
            tmpLeft = keypoints[:, LEFT_KP, :]
            tmpRight = keypoints[:, RIGHT_KP, :]
            keypoints[:, LEFT_KP, :] = tmpRight
            keypoints[:, RIGHT_KP, :] = tmpLeft

        # print keypoints
        return img, masks, keypoints
