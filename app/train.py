from app.dataset import generate_dataset
from app.model import create_model
from keras.api.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from app.settings import amount_of_captchas_for_validation, amount_of_captchas_for_train
from app.logger import logger
from app.settings import epochs, noise
import os

# Генерація датасету
logger.info("Generating training dataset")
X_train, y_train = generate_dataset(amount_of_captchas_for_train, noise)
logger.info("Training dataset generated")

logger.info("Generating validation dataset")
X_val, y_val = generate_dataset(amount_of_captchas_for_validation, noise)
logger.info("Validation dataset generated")

# Ініціалізація моделі
logger.info("Creating model")
model = create_model()

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint(
        "checkpoints/captcha_model_best.keras", save_best_only=True, monitor="val_loss"
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
]

# Навчання
logger.info("Starting model training")

if os.path.exists("app/models/solver.keras"):
    logger.success("Loaded previous model weights")
    model.load_weights("app/models/solver.keras")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=epochs,
    callbacks=callbacks,
)
logger.info("Model training completed")
model.save("app/models/solver.keras")
