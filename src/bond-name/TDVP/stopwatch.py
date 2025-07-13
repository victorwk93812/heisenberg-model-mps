import time
from typing import List, Tuple, Final, Literal

def prGreen(s): print("\033[92m {}\033[00m".format(s))

class stopwatch:
    def __init__(self):
        start_time = time.time()
        self.lap_list: list = [start_time]
    def lap(self, text: str = None, color: Literal["Green"] = "Green"):
        lap_time = time.time()
        self.lap_list.append(lap_time)
        if text != None:
            msg = f"{text}: {self.lap_list[-1] - self.lap_list[-2]}"
            prGreen(msg)
        return self.lap_list[-1] - self.lap_list[-2]