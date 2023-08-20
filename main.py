from torch.utils.data import DataLoader, TensorDataset
from random import shuffle
import numpy as np
import Model
import Validate
import Train
import load_data
import Preprocessing
import splite_data
import torch
import argparse
import Data_aug 
from contextlib import redirect_stdout

if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments with default values
    parser.add_argument('--dataset', type=str, default='S')
    parser.add_argument('--test_ratio', type=float, default=0.9)
    parser.add_argument('--windowSize', type=int, default=5)
    parser.add_argument('--num_pca_components', type=int, default=30)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--aug', type=bool, default=True)
    parser.add_argument('--reduction_ratio', type=int, default=2)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--K_fold', type=int, default=4)
    parser.add_argument('--Test', type=bool, default=True)

    # Parse the arguments
    args = parser.parse_args()

    # Now you can access the arguments using dot notation
    reduction_ratio = args.reduction_ratio
    dataset= args.dataset
    test_ratio= args.test_ratio
    windowSize= args.windowSize
    num_pca_components= args.num_pca_components
    num_epochs= args.num_epochs
    aug = args.aug
    lr= args.lr
    patience= args.patience
    batch_size= args.batch_size
    K_fold = args.K_fold
    train = args.train
    Test = args.Test
    
    # VIT default parameters 
    patch_size = 1
    dropout = 0.1
    mlp_size = 3072
    num_transformer_layers = 12
    num_heads = 2
    dataset_list = ['SA', 'S', 'P', 'PU', 'IP']

    for dataset in dataset_list:
        # load the required dataset                          
        X, y, name = load_data.loadData(dataset)

        print("Data Shape is : ", X.shape)

        # Calculate the number of classes
        num_classes = len(np.unique(y))
        print("Number of Classes:", num_classes)

        unique_classes, counts = np.unique(y, return_counts=True)
        num_classes = len(unique_classes)

        print("Unique Classes:", unique_classes)
        print("Number of Classes:", num_classes)
        print("Class Counts:", counts)

        if dataset == 'SA':
            # Create a mapping dictionary from original labels to new labels
            label_mapping = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}

            # Flatten the y array to make it 1D
            flattened_y = y.ravel()

            # Use the mapping to transform the labels
            new_labels = np.array([label_mapping[label] for label in flattened_y])

            # Reshape the new_labels array to the original shape of y
            original_shape = y.shape
            reshaped_labels = new_labels.reshape(original_shape)
        
            y = reshaped_labels

        # data preprocessing 
        X_normalized = Preprocessing.preprocessing(X)

        # Preprocess the hyperspectral data using PCA and LDA
        reduced_data= Preprocessing.preprocess_hsi_data(X_normalized, y, 
                                                        num_pca_components, 
                                                        num_classes-1)

        # Reconstruct the original 3D hyperspectral image
        rc, n, num_samples = X.shape

        X_reduced = Preprocessing.reconstruct_hsi_data(reduced_data, rc, 
                                                    n, num_classes-1)

        # Get the dimensions of the hyperspectral image data.
        height, width, band = X_reduced.shape
        
        # Apply mirroring to the input hyperspectral image data to augment the dataset.
        mirror_image = Preprocessing.mirror_hsi(height, width, 
                                                band, X_reduced)

        # splite the data
        X_train_positions, X_test_positions, y_train, y_test = splite_data.get_pixel_coordinates(y)

        X_train, X_test = Preprocessing.gain_neighborhood_pixel(mirror_image, X_train_positions, 
                                                                X_test_positions)

        # data augmentation
        if aug:
            aug_data, aug_label= Data_aug.augmentation(X_train,y_train, 0.3)
            #append aug data to original data
            aug_data_arr = np.array(aug_data)
            aug_label_arr = np.array(aug_label)

            # Concatenate the two arrays
            concatenated_data = np.concatenate((X_train, aug_data_arr))
            concatenated_label = np.concatenate((y_train, aug_label_arr))

            ind_list = [i for i in range(len(concatenated_data))]

            shuffle(ind_list)
            
            X_train  = concatenated_data[ind_list, :,:,:]
            y_train = concatenated_label[ind_list,]
        
        # Convert your data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train)
        y_train_tensor = torch.tensor(y_train.reshape(-1))
        X_test_tensor = torch.tensor(X_test)
        y_test_tensor = torch.tensor(y_test.reshape(-1))
        

        # Find the unique classes and their counts
        unique_classes = torch.unique(y_test_tensor)
        num_classes = len(unique_classes)

        print("Number of unique classes:", num_classes)
        print("Train Shape is : ", X_train_tensor.shape)

        # Create TensorDatasets for train and test data
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False)

        img_size = (X_train.shape[1], X_train.shape[2])
        num_bands = X_train.shape[3]
        embedding_dim = X_train.shape[3]

        # Create an instance of the mode
        model = Model.CombinedModel(num_bands, reduction_ratio, num_classes)
        
        # Save the standard output (console) to a file and display in terminal
        with open(f'{name}_Data_results.txt', 'w') as f, redirect_stdout(f):
            if train:
                print("Training ............................................................................")
                # Combine tensors to create a single dataset
                trian_validation_data = TensorDataset(X_train_tensor, y_train_tensor)

                # Create a DataLoader for the combined dataset
                trian_validation_data_loader = DataLoader(trian_validation_data, batch_size=batch_size, shuffle=True)

                Train.k_fold_cross_validation(trian_validation_data_loader, 
                                            num_epochs, 
                                            lr, patience,
                                            batch_size,
                                            K_fold,
                                            num_bands,
                                            reduction_ratio, 
                                            num_classes,
                                            name)   
            if Test:
                print("Testing .............................................................................")
                models = []

                for fold in range(1, K_fold+1):
                    model_filename = f"{name}_models" + "/" + f"trained_model_fold_{fold}.pth"
                    fold_model = Validate.load_model(model, model_filename)
                    models.append(fold_model)

                Validate.evaluate_model(models, test_loader)