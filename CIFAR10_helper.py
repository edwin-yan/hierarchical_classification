from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from common import *


def get_cifar10():
    def create_superclass(x):
        return 0 if x in [0, 1, 8, 9] else 1

    v_create_superclass = np.vectorize(create_superclass)

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    X, Y_sub = np.concatenate((x_train, x_test), axis=0) / 255, np.concatenate((y_train, y_test), axis=0)
    Y_super = v_create_superclass(Y_sub)

    return X, to_categorical(Y_super), to_categorical(Y_sub)


def get_labels(idx, level='super'):
    labels = {
        'super': ['machine', 'animal'],
        'sub': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }
    return labels[level][idx]


v_get_labels = np.vectorize(get_labels)


def plot_sample_images(inputs, v_label_func, top_n=20, n_col=10):
    (images, super_labels_idx, sub_labels_idx) = inputs
    assert len(images) == len(super_labels_idx) == len(sub_labels_idx)
    super_labels, sub_labels = v_label_func(np.argmax(super_labels_idx, axis=1), 'super'), v_label_func(np.argmax(sub_labels_idx, axis=1), 'sub')
    top_n = top_n if top_n < len(images) else len(images)
    n_row = int(top_n / n_col)
    images = images[:top_n]
    figsize = (2.5 * n_col, 2.5 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    for idx, image in enumerate(images):
        ax = axes[idx // n_col, idx % n_col]
        ax.imshow(image)
        ax.set_title(f"{super_labels[idx]} - {sub_labels[idx]}")
    plt.tight_layout()
    plt.show()



def train_and_predict(model, epochs, train, test, batch_size=512):
    (x_train, y_super_train, y_sub_train), (x_test, y_super_test, y_sub_test) = train, test
    model.fit(
        x_train,
        y={"super_output": y_super_train, "sub_output": y_sub_train},
        validation_data=(x_test, {"super_output": y_super_test, "sub_output": y_sub_test}),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
    )

    yhat_super_test, yhat_sub_test = model.predict(x_test, use_multiprocessing=True)
    y_super, yhat_super = v_get_labels(np.argmax(y_super_test, axis=1), 'super'), v_get_labels(np.argmax(yhat_super_test, axis=1), 'super')
    y_sub, yhat_sub = v_get_labels(np.argmax(y_sub_test, axis=1), 'sub'), v_get_labels(np.argmax(yhat_sub_test, axis=1), 'sub')

    # Calculate Metrics
    n_rows = len(x_test)
    error_rate_super = sum(y_super != yhat_super) / n_rows
    error_rate_sub = sum(y_sub != yhat_sub) / n_rows

    hierarchy = gen_hierarchy(y_super_train, y_sub_train)
    n_mismatch = calculate_hiearchy_mismatch(np.argmax(yhat_super_test, axis=1), np.argmax(yhat_sub_test, axis=1), hierarchy)
    mismatch_rate = n_mismatch / n_rows
    print(f"Super Class Error Rate: {error_rate_super}\nSub Class Error Rate: {error_rate_sub}\nMismatch Rate: {mismatch_rate}")
    return error_rate_super, error_rate_sub, mismatch_rate
