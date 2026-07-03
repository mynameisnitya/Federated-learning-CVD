import tensorflow as tf


def create_model():

    model = tf.keras.Sequential([

        tf.keras.Input(shape=(7,)),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dense(1, activation="sigmoid")

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    return model
