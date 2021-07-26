import tensorflow as tf
from config import config


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


def short_offset_loss(short_offset_true, short_offsets_pred, kp_maps_true):
    loss = tf.abs(short_offset_true-short_offsets_pred)/config.KP_RADIUS
    loss = loss*tf_repeat(kp_maps_true, [1, 1, 1, 2])
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(kp_maps_true)+1)
    return loss


def mid_offset_loss(mid_offset_true,mid_offset_pred, kp_maps_true):
    loss = tf.abs(mid_offset_pred-mid_offset_true)/config.KP_RADIUS
    recorded_maps = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        from_kp = edge[0]
        recorded_maps.extend([kp_maps_true[:, :, :, from_kp], kp_maps_true[:, :, :, from_kp]])
    recorded_maps = tf.stack(recorded_maps, axis=-1)
    loss = loss*recorded_maps
    loss = tf.reduce_sum(loss)/(tf.reduce_sum(recorded_maps)+1)
    return loss


def long_offset_loss(long_offset_true, long_offsets_pred, seg_true, crowd_mask, unannotated_mask, overlap_mask):
    loss = tf.abs(long_offsets_pred-long_offset_true)/config.KP_RADIUS
    instances = seg_true*crowd_mask*unannotated_mask*overlap_mask
    loss = loss*instances
    loss = tf.reduce_sum(loss)/(tf.reduce_sum(instances)+1)
    return loss


def segmentation_loss(seg_true, seg_pred, crowd_mask):
    loss = tf.keras.backend.binary_crossentropy(seg_true, seg_pred)
    loss = loss * crowd_mask
    loss = tf.reduce_mean(loss)
    return loss