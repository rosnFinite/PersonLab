import math

import matplotlib
import matplotlib.pyplot as plt
import cv2
from model import get_model
from data_generator import DataGenerator
import numpy as np
from output_processing import *


model = get_model()
model.load_weights("saved_model/model_weights_785.h5")

datagen = DataGenerator(train=False)
sample = datagen.get_one_sample(give_id=785, is_aug=False)
img = np.array([sample[0]])
cv2.imwrite("./outputs/image.png", img[0]*255)

pred = model.predict(img)


H_combined = np.zeros(shape=(400, 400))
H = compute_heatmaps(pred[0][0], pred[1][0])
for i in range(17):
    H[:, :, i] = gaussian_filter(H[:, :, i], sigma=2)
    H_combined = np.add(H[:, :, i]*10, H_combined)
    plt.imsave("outputs/heatmaps/heatmaps"+str(i)+".jpg",  H[:,:,i]*10)
plt.imsave("outputs/heatmaps/combined.jpg", H_combined)

pred_kp = get_keypoints(H)
pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=pred[2][0])
for skel in pred_skels:
    print(skel)
pred_skels = [skel for skel in pred_skels if (skel[:,2]>0).sum() > 4]
print("Number of detected skeletons: {}".format(len(pred_skels)))


img = cv2.imread("./outputs/image.png")
skel_nr = 0
for skel in pred_skels:
    for edge in config.EDGES:
        if skel[edge[0],2] == 0 or skel[edge[1], 2] == 0:
            continue
        x = [int(skel[edge[0], 1]), int(skel[edge[1], 1])]
        y = [int(skel[edge[0], 0]), int(skel[edge[1], 0])]

        if skel_nr == 0:
            img = cv2.line(img, (y[0], x[0]), (y[1], x[1]), color=(255,0,0), thickness=3)
        else:
            img = cv2.line(img, (y[0], x[0]), (y[1], x[1]), color=(0, 255, 0), thickness=3)
    skel_nr += 1

cv2.imwrite("./outputs/image.png", img)
