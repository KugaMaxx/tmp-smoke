from .field import FieldConfig, FieldModel
from .adlstm import ADLSTMConfig, ADLSTM
from .dalle import DALLEConfig, DALLEModel


def prepare_model(model_name: str):
    """
    Prepare and return the specified model with its configuration.
    
    Args:
        model_name (str): Name of the model to prepare. 
                         Options: 'field', 'adlstm', 'dalle'
    
    Returns:
        model: The initialized model instance
    
    Raises:
        ValueError: If the model name is not recognized
    """
    model_name = model_name.lower()

    if model_name == "field":
        model_config = FieldConfig()
        model = FieldModel(model_config)
        return model
    elif model_name == "adlstm":
        model_config = ADLSTMConfig()
        model = ADLSTM(model_config)
        return model
    elif model_name == "dalle":
        model_config = DALLEConfig()
        model = DALLEModel(model_config)
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