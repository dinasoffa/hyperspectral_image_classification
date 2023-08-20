from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import time

# Load a trained model from a file
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    return model

def evaluate_model(models, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
      # Initialize variables for metrics calculation
      start_time = time.time()
      all_predicted_probs = []
      all_labels = []
      all_losses = []
      all_ensemble_predicted = []  # Store the ensemble predictions   

      for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            predicted_probs = []
            model_losses = []

            for model in models:
                model.eval()  # Set the model to evaluation mode
                model.double().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                predicted_probs.append(outputs.softmax(dim=1))
                model_losses.append(loss.item())

            # Combine the predicted probabilities from all models (simple average)
            ensemble_probs = torch.stack(predicted_probs, dim=0).mean(dim=0)
            _, ensemble_predicted = torch.max(ensemble_probs, 1)

            # Collect predictions and labels for evaluation
            all_ensemble_predicted.extend(ensemble_predicted.cpu().numpy())
            all_predicted_probs.extend(ensemble_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_losses.extend(model_losses)
        
    # Calculate average loss
    average_loss = sum(all_losses) / len(all_losses)

    end_time = time.time()
    testing_time = end_time - start_time

    # Calculate evaluation metrics
    report = classification_report(all_labels, all_ensemble_predicted, output_dict=True)
    confusion = confusion_matrix(all_labels, all_ensemble_predicted)
    kappa_acc = cohen_kappa_score(all_labels, all_ensemble_predicted)
    overall_acc = accuracy_score(all_labels, all_ensemble_predicted)

    # Calculate average accuracy manually
    num_classes = len(report) - 3
    class_acc = [report[str(cls)]['recall'] for cls in range(num_classes)]
    class_support = [report[str(cls)]['support'] for cls in range(num_classes)]
    average_acc = sum(acc * support for acc, support in zip(class_acc, class_support)) / sum(class_support)

    print("testing Time (s):", testing_time)
    print("Kappa Accuracy (%):", kappa_acc * 100)
    print("Overall Accuracy (%):", overall_acc * 100)
    print("Average Accuracy (%):", average_acc * 100)
    print("average loss :", average_loss)
    print("Average Accuracy (%):", average_acc * 100)

    # Each accuracy
    print("Each Accuracy (%):")
    for cls in range(num_classes):
        print(f"Class {cls}: {class_acc[cls] * 100}")

    print("Classification Report:")
    print(classification_report(all_labels, all_ensemble_predicted))

    print("Confusion Matrix:")
    print(confusion)

    print("testing finished!")