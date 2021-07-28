import os
#Suppress Tensorflow console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_generator import DataGenerator
from bilinear import refine
from config import config


# Suppress any warning corresponding to matplotlib
warnings.simplefilter(action="ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser(description="Visualizes groundtruth and model prediction")
parser.add_argument("-dataset",
                    dest="dataset",
                    default="validation",
                    help="Determine the dataset from which an image will be taken [validation][train]")
parser.add_argument("-id",
                    dest="image_id",
                    default=None,
                    type=int,
                    help="ID of the image to be visualized. Has to be inside given dataset")
parser.add_argument("-model",
                    dest="model",
                    default=None,
                    help="Name of a model in ./saved_model to be loaded")
parser.add_argument("-save",
                    dest="save",
                    default=False,
                    help="Additional option to save visualization as images")
parser.add_argument("-kp",
                    dest="kp",
                    default=0,
                    type=int,
                    help="Changes visualized kyepoint by given id. [0 - 16]")
args = parser.parse_args()

if args.dataset == "validation":
    datagen = DataGenerator(train=False)
elif args.dataset == "train":
    datagen = DataGenerator()
else:
    print("!! Invalid dataset selected !!")
    print(args.dataset + " was selected but only [validation] or [train] available")

if args.image_id is None:
    gt_data = datagen.get_one_sample(is_aug=False)
elif args.image_id is not None:
    gt_data = datagen.get_one_sample(give_id=args.image_id, is_aug=False)

if args.kp < 0 or args.kp > 16:
    print("!! Keypoint ID is out of range [0-16] !!")
    print("!! Default Keypoint has been chosen !!")
    args.kp = 0

if args.model is None:
    print("Loading model...")
    model = tf.keras.models.load_model("saved_model/model_combined", custom_objects={"refine": refine})
    print("Finished loading model")
elif args.model is not None:
    try:
        print("Loading model...")
        model = tf.keras.models.load_model("saved_model/"+args.model, custom_objects={"refine": refine})
        print("Finished loading model")
    except:
        print("!! Model doesn't exist !!")


prediction = model(np.array([gt_data[0]]), training=False)

fig = plt.figure(figsize=(6,14))
plt.axis("off")
gs = fig.add_gridspec(6, 2, hspace=1)
gs.update(wspace=0.1, hspace=0.2)

#Input
img_ax = plt.subplot(gs[0,0])
plt.tick_params(axis='both', which='both', length=0)
plt.xticks([])
plt.yticks([])
plt.imshow(gt_data[0])

for i in range(5):
    plt.subplot(gs[i+1, 0])
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.imshow(gt_data[i + 1][:, :, args.kp])
    if i == 1 or i == 3:
        plt.imshow(gt_data[i+1][:, :, 2 * args.kp])
    if i == 2:
        # Get Edges containing given keypoint
        edges = [item+(index,) for index, item in enumerate(config.EDGES) if item[0] == args.kp or item[1] == args.kp]
        plt.imshow(gt_data[i+1][:, :, 4*edges[0][2]])
    if i == 4:
        plt.imshow(gt_data[i + 1][:, :, 0])


    plt.subplot(gs[i+1, 1])
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.imshow(prediction[i][0][:, :,args.kp])
    if i == 1 or i == 3:
        plt.imshow(prediction[i][0][:, :, 2 * args.kp])
    if i == 2:
        # Get Edges containing given keypoint
        edges = [item+(index,) for index, item in enumerate(config.EDGES) if item[0] == args.kp or item[1] == args.kp]
        plt.imshow(prediction[i][0][:, :, 4*edges[0][2]])
    if i == 4:
        plt.imshow(prediction[i][0][:, :, 0])


plt.show()
