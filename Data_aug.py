import numpy as np
import cv2

def spectral_band_shuffling(data):
    num_bands = data.shape[2]
    shuffled_data = np.copy(data)

    for i in range(num_bands):
        random_band = np.random.randint(num_bands)
        shuffled_data[:, :, i] = data[:, :, random_band]

    return shuffled_data

def spectral_blurring(data, blur_sigma):
    blurred_data = np.copy(data)

    for i in range(data.shape[2]):
        blurred_data[:, :, i] = cv2.GaussianBlur(data[:, :, i], (0, 0), blur_sigma)

    return blurred_data

def augmentation(data,label,rate):
  aug_data=[]
  aug_label=[]
  for index, i in enumerate(data):
    shuffled=spectral_band_shuffling(i)
    blured =spectral_blurring(i,rate)
    aug_data.append(blured)
    aug_label.append(label[index])

  return aug_data,aug_label