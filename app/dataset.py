import random
from tqdm import tqdm
import numpy as np
from captcha.image import ImageCaptcha
from PIL import ImageDraw, ImageFilter
from keras.api.utils import to_categorical  # Оновлений імпорт
from app.settings import (
    img_width,
    img_height,
    captcha_length,
    char_to_index,
    characters,
)
from app.logger import logger


# Генерація випадкового тексту
def generate_random_text(length):
    text = "".join(random.choices(characters, k=length))
    return text


# Додавання шуму до CAPTCHA
def add_noise(image):
    draw = ImageDraw.Draw(image)

    # Випадкові лінії
    for _ in range(5):
        start_point = (random.randint(0, img_width), random.randint(0, img_height))
        end_point = (random.randint(0, img_width), random.randint(0, img_height))
        draw.line([start_point, end_point], fill=(random.randint(0, 255)), width=1)

    # Випадкові точки
    for _ in range(50):
        x = random.randint(0, img_width)
        y = random.randint(0, img_height)
        draw.point((x, y), fill=(random.randint(0, 255)))

    # Розмиття
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    return image


# Генерація CAPTCHA з шумом
def generate_captcha(text):
    captcha = ImageCaptcha(width=img_width, height=img_height)
    image = captcha.generate_image(text)
    image = image.convert("L")  # Конвертація в ч/б
    return np.array(image)


def generate_captcha_with_noise(text):
    captcha = ImageCaptcha(width=img_width, height=img_height)
    image = captcha.generate_image(text)
    image = image.convert("L")  # Конвертація в ч/б
    image = add_noise(image)
    return np.array(image)


# Генерація датасету
def generate_dataset(num_samples, noise: bool):
    logger.info(f"Starting dataset generation with {num_samples} samples")
    X, y = [], []

    # Використовуємо tqdm для відображення прогресу
    for i in tqdm(range(num_samples), desc="Generating CAPTCHA samples", unit="sample"):
        text = generate_random_text(captcha_length)
        if noise:
            image = generate_captcha_with_noise(text)
        else:
            image = generate_captcha(text)
        X.append(image / 255.0)

        # Перевірка на наявність символів у char_to_index
        try:
            y.append([char_to_index[char] for char in text])
        except KeyError as e:
            logger.warning(
                f"Character '{e.args[0]}' not found in char_to_index for text: {text}"
            )
            continue  # Пропуск цього зразка, якщо є невідомі символи

    logger.info(f"Dataset generation completed")
    X = np.expand_dims(np.array(X), axis=-1)
    y = np.array(y)

    # Перевірка індексів перед використанням to_categorical
    for i in range(captcha_length):
        max_index = len(char_to_index) - 1
        y[:, i] = np.clip(y[:, i], 0, max_index)  # Унеможливлюємо перевищення індексів

    y = [
        to_categorical(y[:, i], num_classes=len(char_to_index))
        for i in range(captcha_length)
    ]
    return X, y
