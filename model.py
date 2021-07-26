import tensorflow as tf
from resnet50 import get_resnet50_base
from config import config
from bilinear import refine


def split_and_refine_mid_offsets(mid_offsets, short_offsets):
    output_mid_offsets = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        to_keypoint = edge[1]
        kp_short_offsets = tf.keras.layers.Lambda(lambda t: t[:, :, :, 2 * to_keypoint:2 * to_keypoint + 2])(
            short_offsets)
        kp_mid_offsets = tf.keras.layers.Lambda(lambda t: t[:, :, :, 2 * mid_idx:2 * mid_idx + 2])(mid_offsets)
        kp_mid_offsets = tf.keras.layers.Lambda(lambda t: refine(t))([kp_mid_offsets, kp_short_offsets])
        output_mid_offsets.append(kp_mid_offsets)

    return tf.keras.layers.Lambda(lambda t: tf.keras.backend.concatenate(t, axis=-1))(output_mid_offsets)


def split_and_refine_long_offsets(long_offsets, short_offsets):
    output_long_offsets = []
    for i in range(config.NUM_KP):
        kp_long_offsets = tf.keras.layers.Lambda(lambda t: t[:, :, :, 2 * i:2 * i + 2])(long_offsets)
        kp_short_offsets = tf.keras.layers.Lambda(lambda t: t[:, :, :, 2 * i:2 * i + 2])(short_offsets)
        refined_1 = tf.keras.layers.Lambda(lambda t: refine(t))([kp_long_offsets, kp_long_offsets])
        refined_2 = tf.keras.layers.Lambda(lambda t: refine(t))([refined_1, kp_short_offsets])
        output_long_offsets.append(refined_2)

    return tf.keras.layers.Lambda(lambda t: tf.keras.backend.concatenate(t, axis=-1))(output_long_offsets)


def get_model(path=None):
    #TODO Stride variabel machen
    if path is None:
        base_model = get_resnet50_base(tf.keras.layers.Input(shape=config.IMG_SHAPE),
                                       output_stride=8,
                                       return_model=True)

        """
        # Only for overfitting on one example 
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = 0.5 
        """

        # Adding output layers for each map

        # Keypoint Maps
        kp_maps = tf.keras.layers.Conv2D(config.NUM_KP, kernel_size=(1, 1), activation="sigmoid")(base_model.output)
        upsample_kp = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(kp_maps)

        # Short-range Offsets
        short = tf.keras.layers.Conv2D(2*config.NUM_KP, kernel_size=(1, 1), activation=None)(base_model.output)
        upsample_short = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(short)

        # Mid-range Offsets
        mid = tf.keras.layers.Conv2D(4*len(config.EDGES), kernel_size=(1, 1), activation=None)(base_model.output)
        upsample_mid = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(mid)
        mid_offsets = split_and_refine_mid_offsets(upsample_mid, upsample_short)

        # Long-range Offsets
        longo = tf.keras.layers.Conv2D(2*config.NUM_KP, kernel_size=(1, 1), activation=None)(base_model.output)
        upsample_longo = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(longo)
        long_offsets = split_and_refine_long_offsets(upsample_longo, upsample_short)

        # Segmentation Mask
        seg = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(base_model.output)
        upsample_seg = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(seg)

        # Combining to one model
        model = tf.keras.models.Model(inputs=base_model.input, outputs=[upsample_kp,
                                                                        upsample_short,
                                                                        mid_offsets,
                                                                        long_offsets,
                                                                        upsample_seg])
        return model
