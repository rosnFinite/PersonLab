from pycocotools.coco import COCO
import sys
import numpy as np
import os
import cv2
from augmentation import AugmentationSelection, Transformer
from groundtruth import get_ground_truth
from config import config

# Path of Annotation file TRAIN
ANNO_TRAIN_FILE = "./coco2017/annotations/person_keypoints_train2017.json"
IMG_TRAIN_DIR = "./coco2017/train2017"

# Path to Annotation file VAL
ANNO_VAL_FILE = "./coco2017/annotations/person_keypoints_val2017.json"
IMG_VAL_DIR = "./coco2017/val2017"


class DataGenerator(object):
    def __init__(self, train=True):
        self.train = train
        if self.train:
            self.coco = COCO(ANNO_TRAIN_FILE)
        else:
            self.coco = COCO(ANNO_VAL_FILE)
        self.img_ids = list(self.coco.imgs.keys())
        self.datasetlen = len(self.img_ids)
        self.id = 0
        self.counter = 1

    def get_one_sample(self, give_id=None, is_aug=True):
        if self.id == self.datasetlen:
            self.id = 0
            self.counter = 1
        if give_id is None:
            img_id = self.img_ids[self.id]
        else:
            img_id = give_id
        try:
            if self.train:
                filepath = os.path.join(IMG_TRAIN_DIR, self.coco.imgs[img_id]["file_name"])
            else:
                filepath = os.path.join(IMG_VAL_DIR, self.coco.imgs[img_id]["file_name"])
        except KeyError:
            print("!! Dataset doesn't contain an image with given id " + str(img_id) + " !!")
            sys.exit()
        img = cv2.imread(filepath)
        h, w, c = img.shape
        crowd_mask = np.zeros((h, w), dtype="bool")
        unannotated_mask = np.zeros((h, w), dtype="bool")
        instance_masks = []
        keypoints = []
        img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        for anno in img_anns:
            # if crowd, don't compute loss
            mask = self.coco.annToMask(anno)
            if anno['iscrowd'] == 1:
                crowd_mask = np.logical_or(crowd_mask, mask)
            # if tiny instance, don't compute loss
            elif anno['num_keypoints'] == 0:
                unannotated_mask = np.logical_or(unannotated_mask, mask)
                instance_masks.append(mask)
                keypoints.append(anno['keypoints'])
            else:
                instance_masks.append(mask)
                keypoints.append(anno['keypoints'])
        if len(instance_masks) <= 0:
            self.id += 1
            return None

        kp = np.reshape(keypoints, (-1, config.NUM_KP, 3))
        instance_masks = np.stack(instance_masks).transpose((1, 2, 0))
        overlap_mask = instance_masks.sum(axis=-1) > 1
        seg_mask = np.logical_or(crowd_mask, np.sum(instance_masks, axis=-1))

        # Augmentation
        single_masks = [seg_mask, unannotated_mask, crowd_mask, overlap_mask]
        all_masks = np.concatenate([np.stack(single_masks, axis=-1), instance_masks], axis=-1)
        if is_aug:
            aug = AugmentationSelection.random()
        else:
            aug = AugmentationSelection.unrandom()
        img, all_masks, kp = Transformer.transform(img, all_masks, kp, aug=aug)

        num_instances = instance_masks.shape[-1]
        instance_masks = all_masks[:, :, -num_instances:]
        seg_mask, unannotated_mask, crowd_mask, overlap_mask = all_masks[:, :, :4].transpose((2, 0, 1))
        seg_mask, unannotated_mask, crowd_mask, overlap_mask = [np.expand_dims(m, axis=-1) for m in [seg_mask,
                                                                                                     unannotated_mask,
                                                                                                     crowd_mask,
                                                                                                     overlap_mask]]

        unannotated_mask = np.logical_not(unannotated_mask)
        crowd_mask = np.logical_not(crowd_mask)
        overlap_mask = np.logical_not(overlap_mask)

        kp = [np.squeeze(k) for k in np.split(kp, kp.shape[0], axis=0)]
        kp_maps, short_offsets, mid_offsets, long_offsets = get_ground_truth(instance_masks, kp)
        self.id += 1
        return [img.astype('float32') / 255.0, kp_maps.astype('float32'), short_offsets.astype('float32'),
                mid_offsets.astype('float32'), long_offsets.astype('float32'), seg_mask.astype('float32'),
                crowd_mask.astype('float32'), unannotated_mask.astype('float32'), overlap_mask.astype('float32')]

    def gen_batch(self, batch_size=4):
        h, w, c = config.IMG_SHAPE
        while True:
            imgs_batch = np.zeros((batch_size, h, w, c))
            kp_maps_batch = np.zeros((batch_size, h, w, config.NUM_KP))
            short_offsets_batch = np.zeros((batch_size, h, w, 2 * config.NUM_KP))
            mid_offsets_batch = np.zeros((batch_size, h, w, 4 * len(config.EDGES)))
            long_offsets_batch = np.zeros((batch_size, h, w, 2 * config.NUM_KP))
            seg_mask_batch = np.zeros((batch_size, h, w, 1))
            crowd_mask_batch = np.zeros((batch_size, h, w, 1))
            unannotated_mask_batch = np.zeros((batch_size, h, w, 1))
            overlap_mask_batch = np.zeros((batch_size, h, w, 1))

            for i in range(batch_size):
                sample = self.get_one_sample()
                while sample is None:  # not to train the images with no instance
                    sample = self.get_one_sample()
                    self.counter += 1
                imgs_batch[i] = sample[0]
                kp_maps_batch[i] = sample[1]
                short_offsets_batch[i] = sample[2]
                mid_offsets_batch[i] = sample[3]
                long_offsets_batch[i] = sample[4]
                seg_mask_batch[i] = sample[5]
                crowd_mask_batch[i] = sample[6]
                unannotated_mask_batch[i] = sample[7]
                overlap_mask_batch[i] = sample[8]
                self.counter += 1

            yield [imgs_batch, kp_maps_batch, short_offsets_batch, mid_offsets_batch, long_offsets_batch,
                   seg_mask_batch, crowd_mask_batch, unannotated_mask_batch, overlap_mask_batch]
