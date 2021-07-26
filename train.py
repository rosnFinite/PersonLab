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

"""
# Loading base model
base_model = get_resnet50_base(tf.keras.layers.Input(shape=(400,400,3)),
                               output_stride = 8,
                               return_model=True)

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
      layer.momentum = 0.5

# adding additional layers
kp_maps = tf.keras.layers.Conv2D(17, kernel_size=(1, 1), activation="sigmoid")(base_model.output)
upsample_kp = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(kp_maps)

short = tf.keras.layers.Conv2D(34, kernel_size=(1,1), activation=None)(base_model.output)
upsample_short = tf.keras.layers.UpSampling2D(size=(8,8), interpolation="bilinear")(short)

mid = tf.keras.layers.Conv2D(4*16, kernel_size=(1,1), activation=None)(base_model.output)
upsample_mid = tf.keras.layers.UpSampling2D(size=(8,8), interpolation="bilinear")(mid)
mid_offsets = split_and_refine_mid_offsets(upsample_mid, upsample_short)

longo = tf.keras.layers.Conv2D(34, kernel_size=(1,1), activation=None)(base_model.output)
upsample_longo = tf.keras.layers.UpSampling2D(size=(8,8), interpolation="bilinear")(longo)
long_offsets = split_and_refine_long_offsets(upsample_longo, upsample_short)

seg = tf.keras.layers.Conv2D(1, kernel_size=(1,1), activation="sigmoid")(base_model.output)
upsample_seg = tf.keras.layers.UpSampling2D(size=(8,8), interpolation="bilinear")(seg)

model = tf.keras.models.Model(inputs=base_model.input, outputs=[upsample_kp,
                                                                upsample_short,
                                                                mid_offsets,
                                                                long_offsets,
                                                                upsample_seg])
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer)
"""

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
    return combined_loss

datagen = DataGenerator()

if config.TRAINING:
    model = get_model()
    # TODO Wert anpassen
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)
    model.summary()
    for epoch in range(config.NUM_EPOCHS):
        # Iterate through complete dataset per epoch
        step = 0

        print("\n** Epoch " + str(epoch))
        for x in range(2000):
            print("\rBatch " + str(x) + "   seen so far: " + str(x * config.BATCH_SIZE), end="")
            # batch = next(datagen.gen_batch(batch_size=BATCH_SIZE))
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

            loss_value = train_step(input_imgs, groundtruth)

            if x % 10 == 0:
                print(
                    "\nTraining loss (for one batch): %.4f   seen so far: %d samples"
                    % (float(loss_value), step * config.BATCH_SIZE)
                )

    model.save("saved_model/model_combined")

else:
    model = tf.keras.models.load_model("saved_model/model_res_momentum", compile=False)
    sample = datagen.get_one_sample(give_id=785, is_aug=False)
    input_img = np.array([sample[0]])
    pred = model.predict(input_img)

    np.save("img.npy", sample[0])
    np.save("kp_maps.npy", sample[1])
    np.save("short_offsets.npy", sample[2])
    np.save("mid_offsets.npy", sample[3])
    np.save("long_offsets.npy", sample[4])
    np.save("seg_mask.npy", sample[5])
    np.save("crowd.npy", sample[6])
    np.save("unannotated.npy", sample[7])
    np.save("overlap_mask.npy", sample[8])

    fig = plt.figure()
    gs = fig.add_gridspec(5, 10, hspace=2)
    gs.update(wspace=0.1, hspace=0.2)

    # Plot original image
    img_ax = plt.subplot(gs[0, 0])
    plt.xticks(())
    plt.yticks(())
    plt.imshow(input_img[0])

    # Plot each predicted heatmap separately
    # x and y correspond to position in figure
    x = 0
    y = 1

    for i in range(16):
        if x == 10:
            y += 1
            x = 0
        # Show predicted Map
        plt.subplot(gs[y, x])
        x += 1
        plt.title(kp_name[i], fontsize=8)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(pred[0][:, :, i])

        # Show original Map
        plt.subplot(gs[y, x])
        x += 1
        plt.title("Org " + kp_name[i], fontsize=8)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(sample[1][:, :, i])

    plt.show()