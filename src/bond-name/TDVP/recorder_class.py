import matplotlib.pyplot as plt
import numpy as np

class Recorder:
    def __init__(self, xlabel="x", ylabel="y", title=""):
        self.x = []
        self.y = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def rec(self, y, x=None):
        """紀錄一筆資料，若只給 y，x 自動為目前長度"""
        if x is None:
            x = len(self.x)
        self.x.append(x)
        self.y.append(y)

    def display(self, limit=10):
        """印出最新幾筆資料"""
        print(f"Recorder ({self.title}) — Last {limit} entries:")
        for i in range(-limit, 0):
            if -i <= len(self.x):
                print(f"  x={self.x[i]}, y={self.y[i]}")

    def plot(self):
        """使用 matplotlib 畫出折線圖"""
        plt.plot(self.x, self.y, marker='o')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.title:
            plt.title(self.title)
        plt.grid(True)
        plt.show()
