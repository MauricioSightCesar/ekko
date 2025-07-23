import logging

class Logger:
    """
    Logger class to handle logging for the application.
    """
    def __init__(self, name, log_file, level=logging.INFO):
        self.logger = self.__setup_logger(name, log_file, level)

    def __setup_logger(self, name, log_file, level):
        """
        Function to setup a logger with a specific name and log file.
        """
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Disable propagation to parent loggers to avoid duplicate logging
        logger.propagate = False

        # Clear existing handlers
        while logger.handlers:
            handler = logger.handlers[0]
            logger.removeHandler(handler)
            handler.close()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)