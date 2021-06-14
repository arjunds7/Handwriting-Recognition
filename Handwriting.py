import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/mnist.npz"


def train_mnist():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            if float(logs.get('accuracy') >= 0.99):
                print("Reached 99% accuracy, so cancelling the training")
                self.model.stop_training = True

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    callbacks = myCallback()
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation='relu'),
                                        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(x_train, y_train, epochs=20, callbacks=[callbacks])
    # model fitting
    return history.epoch, history.history['accuracy'][-1]


train_mnist()
