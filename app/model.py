from keras.api.models import Model
from keras.api.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
)
from keras.api.regularizers import l2
from app.settings import img_height, img_width, captcha_length, num_classes


# Побудова моделі
def create_model():
    input_layer = Input(shape=(img_height, img_width, 1))

    # CNN шари
    x = Conv2D(
        32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.01)
    )(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Вихідні шари
    outputs = [
        Dense(num_classes, activation="softmax")(x) for _ in range(captcha_length)
    ]

    model = Model(inputs=input_layer, outputs=outputs)
    metrics = ["accuracy"] * captcha_length
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=metrics)
    return model
