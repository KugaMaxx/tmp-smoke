from .field import FieldConfig, FieldModel
from .adlstm import ADLSTMConfig, ADLSTM
from .dalle import DALLEConfig, DALLEModel


def prepare_model(model_name: str, config_path: str = None, pretrained_path: str = None):
    """
    Prepare and return the specified model with its configuration.
    """
    model_name = model_name.lower()

    if config_path and pretrained_path:
        raise Warning("Both config_path and pretrained_path are provided. Ignoring config_path.")

    # Define model mapping
    model_mapping = {
        "field": (FieldConfig, FieldModel),
        "adlstm": (ADLSTMConfig, ADLSTM),
        "dalle": (DALLEConfig, DALLEModel)
    }
    
    if model_name in model_mapping:
        config_class, model_class = model_mapping[model_name]
        
        if pretrained_path:
            # Load model from pretrained path
            model = model_class.from_pretrained(pretrained_path)

        elif config_path:
            # Load model configuration from JSON file
            model_config = config_class.from_json_file(config_path)
            model = model_class(model_config)

        else:
            # Initialize model with default configuration
            model_config = config_class()
            model = model_class(model_config)
        
        return model
    else:
        raise ValueError(f"Model {model_name} not recognized.")


# Export all classes and the prepare_model function
__all__ = [
    'FieldConfig', 'FieldModel',
    'ADLSTMConfig', 'ADLSTM', 
    'DALLEConfig', 'DALLEModel',
    'prepare_model'
]