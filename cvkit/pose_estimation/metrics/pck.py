import tensorflow as tf


@tf.function
def distance_map(ref, points):
    return tf.vectorized_map(fn=lambda t: tf.keras.backend.sqrt(1e-9 + tf.reduce_sum(tf.keras.backend.square(ref - t))),
                             elems=points)


def build_pck_metric(n_kps, threshold, per_kp=False):
    assert 0. < threshold < 1.0 and n_kps > 0
    if not per_kp:
        @tf.function
        def pck(y_t, y_p):
            # Expected input dimension batch x n_frames x n_keypoints x 3
            # Generate Mask for padded input
            mask = tf.cast(tf.logical_not(tf.reduce_all(tf.reduce_all(y_t == 0, axis=-1), axis=-1)), dtype=tf.float32)
            # Calculates nxn distance matrix for each frame
            v_t = tf.vectorized_map(
                lambda row: tf.vectorized_map(lambda y: tf.vectorized_map(lambda x: distance_map(x, y), elems=y),
                                              elems=row), elems=y_t)
            # Calculate Maximum distance for each frame and store value based on threshold
            d = tf.reduce_max(tf.reduce_max(v_t, axis=-1), axis=-1, keepdims=True) * threshold
            # Calculate d_i
            y_d = tf.sqrt(tf.reduce_sum(tf.square(y_t - y_p), axis=-1) + 1e-9)
            y_d = tf.cast(y_d < d, dtype=tf.float32)
            y_d = tf.reduce_sum(y_d, axis=-1) * mask

            y_d = y_d / n_kps * 100
            y_d = tf.reduce_sum(y_d, axis=-1) / tf.reduce_sum(mask, axis=-1)
            return tf.reduce_mean(y_d)

        pck._name = f'pck_{int(threshold * 100)}'
    else:
        @tf.function
        def pck(y_t, y_p):
            y_t = tf.cast(y_t, dtype=tf.float32)
            y_p = tf.cast(y_p, dtype=tf.float32)
            # Generate Mask for padded input
            mask = tf.cast(tf.logical_not(tf.reduce_all(tf.reduce_all(y_t == 0, axis=-1), axis=-1, keepdims=True)),
                           dtype=tf.float32)
            # Calculates nxn distance matrix for each frame
            v_t = tf.vectorized_map(
                lambda row: tf.vectorized_map(lambda y: tf.vectorized_map(lambda x: distance_map(x, y), elems=y),
                                              elems=row), elems=y_t)
            # Calculate Maximum distance for each frame and store value based on threshold
            d = tf.reduce_max(tf.reduce_max(v_t, axis=-1), axis=-1, keepdims=True) * threshold
            # Calculate d_i
            y_d = tf.sqrt(tf.reduce_sum(tf.square(y_t - y_p), axis=-1) + 1e-9)
            y_d = tf.cast(y_d < d, dtype=tf.float32) * mask
            y_d = tf.reduce_sum(y_d, axis=-2)  # Axis -2 for per keypoint
            mask = tf.cast(tf.logical_not(tf.reduce_all(tf.reduce_all(y_t == 0, axis=-1), axis=-1)), dtype=tf.float32)
            total = tf.reduce_sum(mask, axis=-1, keepdims=True)
            y_d = y_d / total * 100
            return tf.reduce_mean(y_d, axis=0)

        pck._name = f'per_kp_pck_{int(threshold * 100)}'
    return pck
