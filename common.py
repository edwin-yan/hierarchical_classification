import tensorflow as tf
import numpy as np


def reset_keras(model):
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del model
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def make_multi_output_flow(image_gen, X, y_list, batch_size):
    y_indices = np.arange(y_list[0].shape[0])
    orig_flow = image_gen.flow(X, y=y_indices, batch_size=batch_size)

    while True:
        (X, y_next_i) = next(orig_flow)
        y_next = [y_item[y_next_i] for y_item in y_list]
        yield X, y_next


def calculate_hiearchy_mismatch(super_labels, sub_labels, hierarchy):
    mismatch = 0
    assert len(super_labels) == len(sub_labels)
    for idx, super_label in enumerate(super_labels):
        if sub_labels[idx] not in hierarchy[super_label]:
            mismatch += 1
    return mismatch


def gen_hierarchy(super_label, sub_label, use_argmax=False):
    combined_labels = np.stack((np.argmax(super_label, axis=1), np.argmax(sub_label, axis=1)), axis=1)
    unique_combinations = np.unique(combined_labels, axis=0)
    hierarchy = {}
    for combination in unique_combinations:
        super_class, sub_class = combination
        if super_class in hierarchy.keys():
            hierarchy[super_class].append(sub_class)
        else:
            hierarchy[super_class] = [sub_class]
    return hierarchy
