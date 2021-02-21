import matplotlib.pyplot as plt
import os, sys


def plot(list, path):
    plt.plot(list)
    plt.savefig(path)
    plt.show()
    plt.clf()
    