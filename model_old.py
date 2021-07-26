import tensorflow as tf

IMAGE_SHAPE = (403, 403, 3)
NUM_KP = 17
NUM_EDGES = 16


def resnet50_base():
    resnet = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=IMAGE_SHAPE)
    resnet_output = resnet.output
    """
    a = tf.keras.layers.Conv2D(17, kernel_size=(1, 1), activation="sigmoid")(resnet_output)
    kp_maps = tf.keras.layers.UpSampling2D(size=(31, 31), interpolation="bilinear")(a)
    
    b = tf.keras.layers.Conv2D(2*NUM_KP, kernel_size=(1, 1), activation=None)(resnet_output)
    short_offsets = tf.keras.layers.UpSampling2D(size=(31, 31), interpolation="bilinear")(b)
    """
    c = tf.keras.layers.Conv2D(4*NUM_EDGES, kernel_size=(1, 1), activation=None)(resnet_output)
    mid_offsets = tf.keras.layers.UpSampling2D(size=(31, 31), interpolation="bilinear")(c)
    """
    d = tf.keras.layers.Conv2D(2*NUM_KP, kernel_size=(1, 1), activation=None)(resnet_output)
    long_offsets = tf.keras.layers.UpSampling2D(size=(31, 31), interpolation="bilinear")(d)

    e = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(resnet_output)
    seg_mask = tf.keras.layers.UpSampling2D(size=(31, 31), interpolation="bilinear")(e)
    """
    posenet = tf.keras.models.Model(inputs=resnet.input, outputs=mid_offsets)
    return posenet


def get_model(model_type=0):
    if model_type == 0:
        return resnet50_base()

