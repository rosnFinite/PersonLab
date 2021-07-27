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

datagen = DataGenerator(train=True)

steps_per_epoch = 64115//config.BATCH_SIZE

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
    return [loss_value_kp, loss_value_short, loss_value_mid, loss_value_long, loss_value_seg, combined_loss]


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
        batch = next(datagen.gen_batch(batch_size=config.BATCH_SIZE))
        # batch = datagen.get_one_sample(give_id=785, is_aug=False)
        input_imgs = batch[0].astype("float32")

        groundtruth = {"kp_maps": batch[1].astype("float32"),
                       "short_offsets": batch[2].astype("float32"),
                       "mid_offsets": batch[3].astype("float32"),
                       "long_offsets": batch[4].astype("float32"),
                       "seg_mask": batch[5].astype("float32"),
                       "crowd_mask": batch[6].astype("float32"),
                       "unannotated_mask": batch[7].astype("float32"),
                       "overlap_mask": batch[8].astype("float32")}

        kp_loss, short_loss, mid_loss, long_loss, seg_loss, combined_loss = train_step(input_imgs, groundtruth)

        if x % 10 == 0:
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

model.save("saved_model/model_combined")
