import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def preprocessing(X):
        X_normalize = np.zeros(X.shape)
        for i in range(X.shape[2]):
            X_max = np.max(X[:,:,i])
            X_min = np.min(X[:,:,i])
            X_normalize[:,:,i] = (X[:,:,i] - X_min) / (X_max - X_min)

        return X_normalize

def preprocess_hsi_data(X, y, num_pca_components, num_lda_components):
    # Flatten X and y
    rc, n, num_samples = X.shape
    X_2d = np.reshape(X, (rc * n, num_samples))
    y_1d = y.ravel()

    # Verify that the number of samples in X_2d matches y_1d
    if X_2d.shape[0] != y_1d.shape[0]:
        raise ValueError("Number of samples in X_2d and y_1d must be the same.")

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=num_pca_components)
    XP = pca.fit_transform(X_2d)

    # Determine the maximum allowed number of LDA components
    n_classes = len(np.unique(y_1d))

    #the number of components to be less than the number of classes to avoid collinearity issues.
    max_lda_components = min(XP.shape[1], n_classes - 1)

    # Check if the requested number of LDA components is valid
    if num_lda_components > max_lda_components:
        raise ValueError(f"num_lda_components should be at most {max_lda_components}.")
    # Apply LDA on PCA-processed data
    lda = LinearDiscriminantAnalysis(n_components=num_lda_components)
    XLDA = lda.fit_transform(XP, y_1d)
    return XLDA

def reconstruct_hsi_data(reduced_data, rc, n, num_lda_components):
    # Reshape the reduced data back to the original hyperspectral image shape with desired LDA components
    X_inv = np.reshape(reduced_data, (rc, n, num_lda_components))

    return X_inv

def mirror_hsi(height, width, band, input_normalize, patch = 5):
    # Calculate the padding size as half of the patch size (integer division)
    padding = patch // 2

    # Create a new array 'mirror_hsi' with dimensions (height + 2 * padding, width + 2 * padding, band),
    # filled with zeros, and of data type float. This array will store the mirrored hyperspectral image.
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)

    # Step 1: Copy the original hyperspectral image into the central region of 'mirror_hsi'.
    # This ensures that the original image remains in the center, surrounded by padding.
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize

    # Step 2: Left Mirror
    # Fill the left border of 'mirror_hsi' by mirroring the corresponding columns from the original hyperspectral image.
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]

    # Step 3: Right Mirror
    # Fill the right border of 'mirror_hsi' by mirroring the corresponding columns from the original hyperspectral image.
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]

    # Step 4: Top Mirror
    # Fill the top border of 'mirror_hsi' by mirroring the corresponding rows from the central region of 'mirror_hsi'.
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]

    # Step 5: Bottom Mirror
    # Fill the bottom border of 'mirror_hsi' by mirroring the corresponding rows from the central region of 'mirror_hsi'.
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    # Print some information about the mirror operation
    # print("patch is : {}".format(patch))
    # print("mirror_image shape : [{0}, {1}, {2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))

    # Return the 'mirror_hsi' array, which represents the mirrored hyperspectral image with additional padding.
    return mirror_hsi

def gain_neighborhood_pixel(mirror_image, X_train_positions, X_test_positions, patch = 5):

    '''
    Extract the x and y coordinates of the center point we want to process
    Step 1: Extract the neighborhood patch around the center point from the mirrored image
    This creates a new array 'temp_image' containing the patch
    The indexing 'x:(x+patch)' extracts a patch of size 'patch' in the x-direction
    starting from the x-coordinate 'x'
    The indexing 'y:(y+patch)' extracts a patch of size 'patch' in the y-direction
    starting from the y-coordinate 'y'
    '''
    height, width, bands = mirror_image.shape
    X_train = np.zeros((X_train_positions.shape[0], patch, patch, bands), dtype=float)
    X_test = np.zeros((X_test_positions.shape[0], patch, patch, bands), dtype=float)

    index = 0
    for pos in X_train_positions:
        x_pos = pos[0]
        y_pos = pos[1]
        temp_image = mirror_image[x_pos : (x_pos + patch), y_pos : (y_pos + patch), :]
        X_train[index, :, :, :] = temp_image

        index += 1

    index = 0
    for pos in X_test_positions:
        x_pos = pos[0]
        y_pos = pos[1]
        temp_image = mirror_image[x_pos : (x_pos + patch), y_pos : (y_pos + patch), :]
        X_test[index, :, :, :] = temp_image

        index += 1

    return X_train, X_test