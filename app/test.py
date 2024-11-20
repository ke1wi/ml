import numpy as np
import matplotlib.pyplot as plt
from app.dataset import generate_captcha_with_noise, generate_random_text
from app.model import create_model
from app.settings import index_to_char
from app.logger import logger
from app.settings import captcha_length

# Завантаження моделі
model = create_model()
logger.info("Loading model weights")
model.load_weights("app/models/solver.keras")


# Функція декодування
def decode_prediction(predictions):
    text = ""
    for p in predictions:
        index = np.argmax(p)
        if index in index_to_char:
            text += index_to_char[index]
        else:
            text += "?"  # or some other placeholder for unknown indices
    return text


# Тестування
logger.info("Generating test CAPTCHA")
text_sample = generate_random_text(captcha_length)
image_sample = generate_captcha_with_noise(text_sample)


logger.debug(f"Expected text: {text_sample}")


logger.info("Running model prediction")
image_expanded = np.expand_dims(image_sample / 255.0, axis=(0, -1))
predictions = model.predict(image_expanded)

decoded_text = decode_prediction(predictions)
logger.debug(f"Predicted text: {decoded_text}")

if text_sample == decoded_text:
    logger.success("Prediction matched the expected text!")
else:
    logger.warning("Prediction did not match the expected text")

# Візуалізація
plt.imshow(image_sample, cmap="gray")
plt.title(f"Очікуваний: {text_sample}\nПрогноз: {decoded_text}")
plt.axis("off")
plt.show()
