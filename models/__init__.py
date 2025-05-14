from .mirage_img import *
from .mirage_txt import *

def get_model(config):
    """
    Retrieves the model class specified in the config and initializes it with provided parameters.
    """
    model_name = config['model']['name']
    model_params = config['model'].get('params', {})
    
    # Mapping model names to their classes
    model_classes = {
        "img-linear": ImageLinearModel,
        "cbm-encoder": ObjectClassCBMEncoder,
        "cbm-predictor": ObjectClassCBMPredictor,
        "mirage-img": MiRAGeImg,
        "txt-linear": TextLinearModel,
        "tbm-predictor": TBMPredictor,
        "mirage-txt": MiRAGeTxt
        # Add other models here as needed
    }
    
    if model_name in model_classes:
        model_class = model_classes[model_name]
        return model_class(**model_params)  # Instantiate model with parameters
    else:
        raise ValueError(f"Model {model_name} not recognized. Please check config.")
