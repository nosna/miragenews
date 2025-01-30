import torch
import numpy as np
import json
from tqdm import tqdm
from .metrics import calculate_metrics, find_best_threshold
from models import *

# === SHARED FUNCTIONS ===

def save_model_checkpoint(model, save_path, threshold):
    """
    Save the model state and threshold as a checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path to save the checkpoint.
        threshold (float): The best threshold to save.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_threshold": threshold,
    }
    torch.save(checkpoint, save_path)


def load_model_checkpoint(model, save_path):
    """
    Load the model state and best threshold from a checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        save_path (str): Path to the checkpoint.
        
    Returns:
        model (torch.nn.Module): Model loaded with state_dict.
        best_threshold (float): Best threshold saved in the checkpoint.
    """
    checkpoint = torch.load(save_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["best_threshold"] if "best_threshold" in checkpoint else None
    return model, best_threshold


def evaluate_model(model, data_loader, criterion, device="cuda", threshold=0.5, cbm_encoder=None, concept_num=300):
    """
    Evaluate the model on a given dataset, calculating metrics using a specified threshold.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.
        criterion: Loss function.
        device (str): Device for computation.
        threshold (float): Threshold for binary classification.
        cbm_encoder (torch.nn.Module, optional): CBM encoder model, if using CBM Predictor.
        concept_num (int): Number of concepts in CBM Predictor.
        
    Returns:
        avg_loss (float): Average loss over the dataset.
        metrics (dict): Calculated metrics using the specified threshold.
    """
    model.eval()
    y_true = []
    y_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass logic based on model type
            if isinstance(model, ObjectClassCBMEncoder):
                classifier_idx = 0
                outputs = model(inputs.float(), classifier_idx)
            # elif isinstance(model, ObjectClassCBMPredictor) and cbm_encoder:
            #     concept_features = [cbm_encoder(inputs.float(), i) for i in range(concept_num)]
            #     pred_scores = torch.cat(concept_features, dim=1)
            #     outputs = model(pred_scores.to(device))
            else:
                outputs = model(inputs.float())

            # Calculate loss and store probabilities

            loss = criterion(outputs.squeeze(-1), labels.float())
            total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(outputs.squeeze(-1).cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    metrics = calculate_metrics(y_true, y_probs, threshold)

    model.train()
    return avg_loss, metrics, y_true, y_probs  # Return y_true and y_probs for threshold finding

def save_metrics(metrics, save_path):
    """
    Save metrics as a JSON file.

    Args:
        metrics (dict): Dictionary of metrics to save.
        save_path (str): Path to the file where metrics will be saved.
    """
    with open(save_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

# === TRAINING FUNCTION ===

def train_model(model, train_loader, eval_loader, config, classifier_idx=None, cbm_encoder=None, device="cuda"):
    """
    Train the model with early stopping and save the best model checkpoint with the threshold.
    """
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    best_val_loss = float('inf')
    patience_counter = 0
    best_threshold = 0.5

    for epoch in tqdm(range(config['training']['epochs']), desc="Training"):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if isinstance(model, ObjectClassCBMEncoder):
                outputs = model(inputs.float(), classifier_idx)
            # elif isinstance(model, ObjectClassCBMPredictor) and cbm_encoder:
            #     concept_features = [cbm_encoder(inputs.float(), i) for i in range(config.get('concept_num', 300))]
            #     pred_scores = torch.cat(concept_features, dim=1)
            #     outputs = model(pred_scores.to(device))
            else:
                outputs = model(inputs.float())

            loss = criterion(outputs.squeeze(-1), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        # print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Training Loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        eval_loss, val_metrics, y_true, y_probs = evaluate_model(model, eval_loader, criterion, device, cbm_encoder=cbm_encoder)
        # print(f"Validation Loss: {eval_loss:.4f}, Metrics: {val_metrics}")
        
        # Find and save the best threshold based on validation
        best_threshold = find_best_threshold(y_true, y_probs)
        # print(f"Best Threshold Found: {best_threshold}")

        if eval_loss < best_val_loss - 0.001:
            best_val_loss = eval_loss
            patience_counter = 0
            save_model_checkpoint(model, config['training']['save_path'], best_threshold)
        else:
            patience_counter += 1
        
        if patience_counter > 10:
            print("Early stopping triggered. Training completed.")
            break


# === TESTING FUNCTION ===

def test_model(model, test_loader, checkpoint_path, device="cuda", model_2=None):
    """
    Load the best model and threshold, then evaluate on the test set and save metrics.
    
    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        checkpoint_path (str): Path to the saved model checkpoint.
        device (str): Device for computation (e.g., 'cuda' or 'cpu').

    Returns:
        dict: Test metrics.
    """
    criterion = torch.nn.BCELoss()
    # Load model and best threshold
    model, best_threshold = load_model_checkpoint(model, checkpoint_path)
    # Evaluate on the test set with the loaded threshold
    test_loss, test_metrics, _, _ = evaluate_model(model, test_loader, criterion, device, threshold=best_threshold)
    # print("Test Metrics:", test_metrics)

    return test_metrics

def test_multimodal_model(image_model, text_model, test_loader, threshold=0.5, device='cuda'):
    image_model.eval()
    text_model.eval()
    criterion = torch.nn.BCELoss()
    y_true = []
    y_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for image_inputs, text_inputs, labels in test_loader:
            image_inputs, text_inputs, labels = image_inputs.to(device), text_inputs.to(device), labels.to(device)

            # Forward pass logic based on model type
            if isinstance(image_model, ObjectClassCBMEncoder):
                classifier_idx = 0
                image_outputs = image_model(image_inputs.float(), classifier_idx)
            # elif isinstance(model, ObjectClassCBMPredictor) and cbm_encoder:
            #     concept_features = [cbm_encoder(inputs.float(), i) for i in range(concept_num)]
            #     pred_scores = torch.cat(concept_features, dim=1)
            #     outputs = model(pred_scores.to(device))
            else:
                image_outputs = image_model(image_inputs.float())

            text_outputs = text_model(text_inputs.float())
            # Calculate loss and store probabilities
            outputs = (image_outputs + text_outputs) / 2
            loss = criterion(outputs.squeeze(-1), labels.float())
            total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(outputs.squeeze(-1).cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    metrics = calculate_metrics(y_true, y_probs, threshold)

    return metrics