"""
File: one_hot_encode.py

Encode mask as one hot code.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


if __name__ == '__main__':
    # Load mask
    single_mask_path = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/mask/DF1_210-1213_flip-lr.png'
    mask = Image.open(single_mask_path)

    # Preprocessing
    mask_arr = np.asarray(mask)
    mask = np.expand_dims(mask_arr, axis=2)
    print(mask.shape)

    # one hot encode
    palette = [[0], [1], [2], [3], [4]]
    result = mask_to_onehot(mask, palette)

    # Plot result
    print(result.shape)
    for i in range(result.shape[2]):
        plt.imshow(result[:, :, i].squeeze())
        plt.show()
