from config import Config
from tensorboardX import SummaryWriter
import os
from time import time


class Logger:

    def __init__(self):
        self.writer = SummaryWriter(log_dir=os.path.join("./data", "logs", str(int(time()))))

    @staticmethod
    def log_info(msg: str):
        print(msg)

    def scalar(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def histogram(self, key, values, step):
        if len(values) > 0:
            self.writer.add_histogram(key, values, step)
