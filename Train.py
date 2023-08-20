from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import torch
import time
import Model
import os

def reset_parameters(num_bands, reduction_ratio, num_classes):
    # Create an instance of the mode
    model = Model.CombinedModel(num_bands, reduction_ratio, num_classes)

    return model

def k_fold_cross_validation(data_loader, num_epochs, learning_rate, 
                            patience, batch_size, num_folds, 
                            num_bands, reduction_ratio, 
                            num_classes, name):
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_loader.dataset.tensors[0], data_loader.dataset.tensors[1])):
        print(f"Fold {fold+1}/{num_folds}:")

        train_dataset = Subset(data_loader.dataset, train_idx)
        val_dataset = Subset(data_loader.dataset, val_idx)

        train_loader_fold = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = reset_parameters(num_bands, reduction_ratio, num_classes)  # Reset model parameters before each fold
        model = train_model(model.double(), train_loader_fold, val_loader_fold, num_epochs, learning_rate, patience)

        # Save the model with a different name for each fold
        # Create the new directory if it doesn't exist
        if not os.path.exists(f"{name}_models"):
            os.makedirs(f"{name}_models")

        model_filename = f"trained_model_fold_{fold+1}.pth"
        torch.save(model.state_dict(), f"{name}_models" + "/" + model_filename)
        
        print("Trained model saved to:", f"{name}_models" + "/" + model_filename)
        print("Training for this fold finished!")

    print("Training for all folds finished!")

def train_model(model, train_loader, val_loader, 
                num_epochs, learning_rate, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Move model to the device
    model.to(device)

    # Initialize early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Initialize variables for metrics calculation
    start_time = time.time()
    all_predicted = []
    all_labels = []

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Collect predictions and labels for evaluation
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Print the average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * total_correct / total_samples

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_samples += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = 100 * val_correct / val_samples

        # Get the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%, "
              f"Current Learning Rate: {current_lr}")

        scheduler.step(epoch_loss)

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping! Validation loss did not improve for {} epochs.".format(patience))
                break

    end_time = time.time()
    training_time = end_time - start_time

    # Calculate evaluation metrics
    report = classification_report(all_labels, all_predicted, output_dict=True)
    confusion = confusion_matrix(all_labels, all_predicted)
    kappa_acc = cohen_kappa_score(all_labels, all_predicted)
    overall_acc = accuracy_score(all_labels, all_predicted)

    # Calculate average accuracy manually
    num_classes = len(report) - 3
    class_acc = [report[str(cls)]['recall'] for cls in range(num_classes)]
    class_support = [report[str(cls)]['support'] for cls in range(num_classes)]
    average_acc = sum(acc * support for acc, support in zip(class_acc, class_support)) / sum(class_support)

    print("Training Time (s):", training_time)
    print("Kappa Accuracy (%):", kappa_acc * 100)
    print("Overall Accuracy (%):", overall_acc * 100)
    print("Average Accuracy (%):", average_acc * 100)

    # Each accuracy
    print("Each Accuracy (%):")
    for cls in range(num_classes):
        print(f"Class {cls}: {class_acc[cls] * 100}")

    print("Classification Report:")
    print(classification_report(all_labels, all_predicted))

    print("Confusion Matrix:")
    print(confusion)

    print("Training finished!")
    
    return model