import tensorflow as tf
from data_generator import DataGenerator
import model
import numpy as np
import matplotlib.pyplot as plt
import wandb


NUM_EPOCHS = 1
KP_RADIUS = 32
BATCH_SIZE = 8
TRAIN = False
kp_name = ["Nase",
           "Schulter(r)",
           "Ellenbogen(r)",
           "Handgelenk(r)",
           "Schulter(l)",
           "Ellenbogen(l)",
           "Handgelenk(l)",
           "Hüfte(r)",
           "Knie(r)",
           "Fußgelenk(r)",
           "Hüfte(l)",
           "Knie(l)",
           "Fußgelenk(l)",
           "Auge(r)",
           "Auge(l)",
           "Ohr(r)",
           "Ohr(l)"]

datagen = DataGenerator()
print(datagen.datasetlen)


def tf_repeat(tensor, repeats):
    """
    From  https://github.com/tensorflow/tensorflow/issues/8246

    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.compat.v1.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor


def hm_loss(hm_gt, hm_pred, crowd_mask, unannotated_mask):
    binary_loss = tf.keras.backend.binary_crossentropy(hm_gt, hm_pred, from_logits=True)
    masked_loss = binary_loss * crowd_mask * unannotated_mask
    loss = tf.reduce_mean(masked_loss)
    return loss


def short_offset_loss(short_offset_true,short_offsets_pred,kp_maps_true):
    loss = tf.abs(short_offset_true-short_offsets_pred)/KP_RADIUS
    loss = loss*tf_repeat(kp_maps_true, [1, 1, 1, 2])
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(kp_maps_true)+1)
    return loss # *config.LOSS_WEIGHTS['short']


model = model.get_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=hm_loss, optimizer=optimizer, metrics=[tf.keras.metrics.binary_accuracy])

if TRAIN:
    #wandb.login(key="e0ab6f77ccb17acb611cc071af502e08285be067")
    #wandb.init(project="posenet")
    for epoch in range(NUM_EPOCHS):
        # Iterate through complete dataset per epoch
        step = 0
        """
        while datagen.counter < datagen.datasetlen - BATCH_SIZE:
            print("-- Next Batch Generated")
            batch = next(datagen.gen_batch(batch_size=BATCH_SIZE))
    
            input_imgs = batch[0].astype("float32")
            kp_maps_gt = batch[1].astype("float32")
            short_offsets_gt = batch[2].astype("float32")
            mid_offsets_gt = batch[3].astype("float32")
            long_offsets_gt = batch[4].astype("float32")
            seg_mask_gt = batch[5].astype("float32")
            crowd_mask_gt = batch[6].astype("float32")
            unannotated_mask_gt = batch[7].astype("float32")
            overlap_mask_gt = batch[8].astype("float32")
        """
        # Nur für ein Datensatz
        for x in range(100):
            # print(x)
            sample = datagen.get_one_sample(give_id=36, is_aug=True)
            input_imgs = np.array([sample[0]])
            kp_maps_gt = np.array([sample[1]])
            short_offsets_gt = np.array([sample[2]])
            mid_offsets_gt = np.array([sample[3]])
            long_offsets_gt = np.array([sample[4]])
            seg_mask_gt = np.array([sample[5]])
            crowd_mask_gt = np.array([sample[6]])
            unannotated_mask_gt = np.array([sample[7]])
            overlap_mask_gt = np.array([sample[8]])

            with tf.GradientTape() as tape:
                logits = model(input_imgs, training=True)
                loss_value = hm_loss(kp_maps_gt, logits, crowd_mask_gt, unannotated_mask_gt)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if x % 20 == 0:
                #wandb.log({"kp_map_loss": float(loss_value)})
                print(
                    "Training loss (for one batch): %.4f   seen so far: %d samples"
                    % (float(loss_value), step * BATCH_SIZE)
                )


    model.save("saved_model/model_1")
else:
    model = tf.keras.models.load_model("saved_model/model_1", compile=False)
    sample = datagen.get_one_sample(give_id=36, is_aug=False)
    input_img = np.array([sample[0]])
    pred = model.predict(input_img)

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

    for i in range(17):
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








