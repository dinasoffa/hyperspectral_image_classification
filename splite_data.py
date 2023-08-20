import numpy as np
from sklearn.model_selection import train_test_split

def get_pixel_coordinates(labels):
    width, height = labels.shape

    pixel_coordinates = []
    y_data = []
    for x in range(width):
        col = []
        for y in range(height):
            if labels[x,y] > 0:
                pixel_coordinates.append((x, y))
                y_data.append(labels[x,y])
    pixel_coordinates = np.array(pixel_coordinates)
    y_data = np.array(y_data) - 1

    X_train_positions, X_test_positions, y_train, y_test = train_test_split(pixel_coordinates, y_data, test_size= 0.9 , stratify= y_data, random_state= 42)

    return X_train_positions, X_test_positions, y_train, y_test