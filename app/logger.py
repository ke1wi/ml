from loguru import logger
import sys

# Додавання виведення логів у термінал
logger.add(sys.stderr, level="INFO")  # Виводити повідомлення рівня INFO і вище
