import string

# Налаштування CAPTCHA
img_width, img_height = 160, 60  # Розмір CAPTCHA
captcha_length = 5  # Кількість символів
characters = string.ascii_letters + string.digits + "абвгдеёжзийклмнопрстуфхцчшщъьюя"
characters = characters.lower() + characters.upper()
char_to_index = {char: idx for idx, char in enumerate(characters)}  # Символ у індекс
index_to_char = {idx: char for char, idx in char_to_index.items()}  # Індекс у символ
num_classes = len(char_to_index)
epochs = 10
amount_of_captchas_for_train = 5000
amount_of_captchas_for_validation = 500
noise = False
