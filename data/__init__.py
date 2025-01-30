from .dataset import *
import yaml
from PIL import Image
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_dataset(config, is_eval=False, test_set=None):
    """
    Retrieves the dataset class specified in the config and initializes it with provided parameters.

    Args:
        config (dict): Configuration dictionary loaded from a YAML file.
        is_eval (bool): Flag indicating whether to load an evaluation (validation) dataset.
        test_set (str, optional): Specifies the test dataset to load. Use 'test1', 'test2', etc.,
                                      for individual test sets, 'all' for all test sets, or None for train/validation.

    Returns:
        Tuple[Dataset, str] or List[Tuple[Dataset, str]]: Returns (dataset instance, test_name) for a single dataset,
                                                          or a list of (dataset instance, test_name) tuples for all test sets.
    """
    # Mapping dataset names to their classes
    dataset_classes = {
        "img-or-text": MiRAGeImageOrTextDataset,
        "multimodal": MiRAGeNewsDataset,
        # Add other dataset classes here if needed
    }

    # Handle loading of all test datasets if specified
    if test_set == "all":
        return [
            (dataset_classes[config[key]['name']](**config[key]['params']), config[key]['test_name'])
            for key in config if key.startswith("test") and key != "testing" and config[key]['name'] in dataset_classes 
        ]

    # Determine the appropriate dataset section in config
    dataset_key = test_set if test_set else ('val_dataset' if is_eval else 'train_dataset')
    
    # Retrieve dataset name, parameters, and test name (if applicable)
    dataset_name = config[dataset_key]['name']
    dataset_params = config[dataset_key]['params']
    test_name = config[dataset_key].get('test_name', '')

    # Initialize and return the dataset and test name
    if dataset_name in dataset_classes:
        dataset_class = dataset_classes[dataset_name]
        return dataset_class(**dataset_params), test_name
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Please check the config file.")
    

def get_object_class(class_file_path='data/class_names.txt', limit=300):
    """
    Loads object class names from a text file.

    Args:
        class_file_path (str): Path to the text file containing class names, with each class name on a separate line.
        limit (int, optional): The maximum number of classes to load. If None, loads all classes.

    Returns:
        List[str]: A list of class names.
    """
    classes = []
    with open(class_file_path, 'r', encoding='utf-8') as file:
        classes = [line.strip() for line in file][:limit]
    return classes

def get_object_class_caption(class_file_path='data/class_names.txt', limit=300):
    """
    Loads object class captions from a text file.

    Args:
        class_file_path (str): Path to the text file containing class names, with each class name on a separate line.
        limit (int, optional): The maximum number of classes to load. If None, loads all classes.

    Returns:
        List[str]: A list of class captions.
    """
    classes = []
    with open(class_file_path, 'r', encoding='utf-8') as file:
        classes = [f'a photo of {line.strip().lower()}' for line in file][:limit]
    return classes


def get_object_class_config(class_name):
    config = {
        'train_dataset': {
            'name': 'img-or-text',
            'params': {
                'real_pt': f"encodings/crops/{class_name}/train/real.pt",
                'fake_pt': f"encodings/crops/{class_name}/train/fake.pt",
            }
        },
        'val_dataset': {
            'name': 'img-or-text',
            'params': {
                'real_pt': f"encodings/crops/{class_name}/validation/real.pt",
                'fake_pt': f"encodings/crops/{class_name}/validation/fake.pt",
            }
        }
    }
    return config

def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image