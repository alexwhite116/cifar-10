import numpy as np
import pandas
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import qbstyles
import cv2

qbstyles.mpl_style(dark=True)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

TRAINING1 = unpickle('cifar-10-batches-py/data_batch_1')
TRAINING2 = unpickle('cifar-10-batches-py/data_batch_2')
TRAINING3 = unpickle('cifar-10-batches-py/data_batch_3')
TRAINING4 = unpickle('cifar-10-batches-py/data_batch_4')
TRAINING5 = unpickle('cifar-10-batches-py/data_batch_5')
TEST = unpickle('cifar-10-batches-py/test_batch')

def imageFromSet(dataset, index=0):
    redRows = [dataset[b'data'][index][i:i+32] for i in range(0, 1024, 32)]
    greenRows = [dataset[b'data'][index][i:i+32] for i in range(1024, 2048, 32)]
    blueRows = [dataset[b'data'][index][i:i+32] for i in range(2048, 3072, 32)]
    img = [[[] for i in range(32)] for i in range(32)]
    for i in range(32):
        for j in range(32):
            img[j][i] = [redRows[j][i], greenRows[j][i], blueRows[j][i]]
    return np.array(img)

cv2.imwrite('img.png', imageFromSet(TRAINING1))
cv2.imshow('img', imageFromSet(TRAINING1))
cv2.waitKey(0)