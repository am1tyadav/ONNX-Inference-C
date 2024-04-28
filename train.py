import keras
import tensorflow as tf


def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28), name="model_input"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax", name="model_output"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    _ = model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=5
    )

    tf.saved_model.save(model, "saved_model")


if __name__ == "__main__":
    train()
