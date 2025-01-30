import argparse
from models import get_model
from data import load_config, get_dataset, get_object_class, get_object_class_config
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import train_model
import torch
from tqdm import tqdm

def create_balanced_sampler(dataset):
    # Calculate the class distribution and assign weights inversely proportional to class frequency
    class_counts = torch.bincount(dataset.labels.long())
    if (class_counts == 0).any():
        print(class_counts)
        return None
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[int(label)] for label in dataset.labels]
    
    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def main(mode, model_class):
    torch.manual_seed(42)
    
    # Load configuration
    config = load_config(f"configs/{mode}/{model_class}.yaml")
    print(f"Loaded configuration for {model_class}")

    # Set device and batch size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config['training']['batch_size']

    # Initialize model
    model = get_model(config).to(device)
    print(f"Loaded model: {model}")

    if model_class == "cbm-encoder":
        # Special case for cbm-encoder: iterate through object classes and create separate loaders for each
        classes = get_object_class()

        for i, class_name in enumerate(tqdm(classes, desc=f"Encoding object classes")):
            object_class_config = get_object_class_config(class_name)
            object_class_config['training'] = config['training']
            try:
                # Train dataset and loader for current subdirectory
                train_dataset, _ = get_dataset(object_class_config, is_eval=False)
                sampler = create_balanced_sampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

                # Evaluation dataset and loader for current subdirectory
                eval_dataset, _ = get_dataset(object_class_config, is_eval=True)
                sampler = create_balanced_sampler(eval_dataset)
                eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
            except:
                continue

            # Train model for current subdirectory
            train_model(model, train_loader, eval_loader, object_class_config, classifier_idx=i)
    
    else:
        # Default case: load dataset and data loaders once based on config
        print("Loading datasets...")
        
        # Initialize training dataset and data loader
        train_dataset, _ = get_dataset(config, is_eval=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize evaluation dataset and data loader
        eval_dataset, _ = get_dataset(config, is_eval=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        
        # Train model
        train_model(model, train_loader, eval_loader, config)
    
    print("Training completed successfully.")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model with specified configuration")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['image', 'text', 'multimodal'],
        required=True,
        help="Specify model mode: 'image', 'text', or 'multimodal'"
    )
    parser.add_argument(
        '--model_class',
        type=str,
        required=True,
        help="Choose from 'linear', 'cbm-encoder', 'cbm-predictor', 'mirage', or provide a custom model class"
    )
    
    args = parser.parse_args()
    main(args.mode, args.model_class)
