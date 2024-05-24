from loguru import logger
import sys

# Define ANSI color codes directly in the format string
prefix_color = "\033[95m"  # Light Magenta for prefix (TRILL, date, level)
info_color = "\033[1;37m"  # Bolded White for INFO messages
warning_color = "\033[1;33m"  # Bolded Orange for WARNING messages
error_color = "\033[1;31m"  # Bolded Red for ERROR messages
reset_color = "\033[0m"  # Resets the color to default

# Define a function to select the message color based on the log level
def get_message_color(level):
    if level == "INFO":
        return info_color
    elif level == "WARNING":
        return warning_color
    elif level == "ERROR":
        return error_color
    else:
        return info_color

# Custom format function that incorporates colors
def format_record(record):
    level = record["level"].name
    message_color = get_message_color(level)
    return f"{prefix_color}(TRILL) {record['time']:YYYY-MM-DD HH:mm:ss} | {message_color}{record['level']} {reset_color}{prefix_color}| {message_color}{record['message']}{reset_color}\n"

def setup_logger(file_path):
    logger.remove()  # Remove default handlers

    # Console handler with custom formatting
    logger.add(sys.stdout, format=format_record, level="DEBUG")

    # File handler without color formatting
    file_format = "(TRILL) {time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n"
    logger.add(file_path, format=file_format, colorize=False, level="INFO")
