from datetime import datetime

from config import Config


class AppLogger:
    def __init__(self):
        Config.LOGS_PATH.mkdir(parents=True, exist_ok=True)

    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n"
        )
