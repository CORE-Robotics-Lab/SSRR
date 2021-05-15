import os
import tensorflow as tf
import numpy as np
import random


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_video(ims, filename, fps=30.0):
    import cv2
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()
