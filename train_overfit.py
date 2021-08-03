import tensorflow as tf
from data_generator import DataGenerator
from bilinear import bilinear_sampler
import numpy as np
import matplotlib.pyplot as plt
import wandb
from model import get_model
from resnet50 import get_resnet50_base
from loss_functions import hm_loss, short_offset_loss, mid_offset_loss, long_offset_loss, segmentation_loss
from config import config

datagen = DataGenerator(train=False)

steps_per_epoch = 64115//config.BATCH_SIZE
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
"""


def show_hm(gt, pred):
    fig_hm = plt.figure(figsize=(14, 6))
    plt.axis("off")
    gs = fig_hm.add_gridspec(4, 10, hspace=1)
    gs.update(wspace=0.1, hspace=0.2)

    x = 0
    y = 0
    for i in range(config.NUM_KP):
        if x < 10:
            plt.subplot(gs[y, x])
            plt.tick_params(axis='both', which='both', length=0)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(pred[0][:, :, i])

            plt.subplot(gs[y, x+1])
            plt.tick_params(axis='both', which='both', length=0)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(gt[1][:, :, i])
            x += 2
        else:
            x = 0
            y += 1
    plt.show()

@tf.function
def train_step(img, y):
    with tf.GradientTape() as tape:
        kp_pred, short_pred, mid_pred, long_pred, seg_pred = model(img, training=True)
        loss_value_kp = hm_loss(y["kp_maps"], kp_pred, y["crowd_mask"], y["unannotated_mask"])
        loss_value_short = short_offset_loss(y["short_offsets"], short_pred, y["kp_maps"])
        loss_value_mid = mid_offset_loss(y["mid_offsets"], mid_pred, y["kp_maps"])
        loss_value_long = long_offset_loss(y["long_offsets"], long_pred, y["seg_mask"], y["crowd_mask"],
                                           y["unannotated_mask"], y["overlap_mask"])
        loss_value_seg = segmentation_loss(y["seg_mask"], seg_pred, y["crowd_mask"])
        combined_loss = 4*loss_value_kp + loss_value_short + 0.4*loss_value_mid + 0.1*loss_value_long + 2*loss_value_seg
    grads = tape.gradient(combined_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return [loss_value_kp, loss_value_short, loss_value_mid, loss_value_long, loss_value_seg, combined_loss, kp_pred]


model = get_model()
# TODO Wert anpassen
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer)
model.summary()
for epoch in range(config.NUM_EPOCHS):
    # Iterate through complete dataset per epoch
    step = 0

    print("\n** Epoch " + str(epoch))
    for x in range(steps_per_epoch):
        print("\rBatch " + str(x)+"/"+str(steps_per_epoch) + "   seen so far: " + str(x * config.BATCH_SIZE), end="")
        # batch = next(datagen.gen_batch(batch_size=config.BATCH_SIZE))
        batch = datagen.get_one_sample(give_id=785, is_aug=False)
        input_imgs = np.array([batch[0].astype("float32")])

        groundtruth = {"kp_maps": np.array([batch[1].astype("float32")]),
                       "short_offsets": np.array([batch[2].astype("float32")]),
                       "mid_offsets": np.array([batch[3].astype("float32")]),
                       "long_offsets": np.array([batch[4].astype("float32")]),
                       "seg_mask": np.array([batch[5].astype("float32")]),
                       "crowd_mask": np.array([batch[6].astype("float32")]),
                       "unannotated_mask": np.array([batch[7].astype("float32")]),
                       "overlap_mask": np.array([batch[8].astype("float32")])}

        kp_loss, short_loss, mid_loss, long_loss, seg_loss, combined_loss, pred = train_step(input_imgs, groundtruth)

        if x % 100 == 0:
            print(
                "\nCombined Training loss (for one batch): %.4f   seen so far: %d samples"
                % (float(combined_loss), step * config.BATCH_SIZE)
            )
            print(
                "-- Heatmap Loss : %.4f" % (float(kp_loss))
            )
            print(
                "-- Short Loss : %.4f " % (float(short_loss))
            )
            print(
                "-- Mid loss : %.4f " % (float(mid_loss))
            )
            print(
                "-- Long Loss : %.4f " % (float(mid_loss))
            )
            print(
                "-- Segmentation Loss : %.4f " % (float(seg_loss))
            )
            show_hm(gt=batch, pred=pred)
        if x % 1000 == 0:
            model.save_weights("saved_model/model_weights_785.h5")





