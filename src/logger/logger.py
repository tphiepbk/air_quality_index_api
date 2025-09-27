import logging

color = {
    "blue": '\x1b[38;5;39m',
    "yellow": '\x1b[38;5;226m',
    "red": '\x1b[38;5;196m',
    "reset": '\x1b[0m'
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def warning(message, *args):
    logging.warning(message.format(*args))

def info(message, *args):
    logging.info(message.format(*args))

def debug(message, *args):
    logging.debug(message.format(*args))

