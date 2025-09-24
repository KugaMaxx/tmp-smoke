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

    if model_name == "field":
        if pretrained_path:
            model = FieldModel.from_pretrained(pretrained_path)
        elif config_path:
            model_config = FieldConfig.from_json_file(config_path)
            model = FieldModel(model_config)
        else:
            model_config = FieldConfig()
            model = FieldModel(model_config)
        return model
    elif model_name == "adlstm":
        if pretrained_path:
            model = ADLSTM.from_pretrained(pretrained_path)
        elif config_path:
            model_config = ADLSTMConfig.from_json_file(config_path)
            model = ADLSTM(model_config)
        else:
            model_config = ADLSTMConfig()
            model = ADLSTM(model_config)
        return model
    elif model_name == "dalle":
        if pretrained_path:
            model = DALLEModel.from_pretrained(pretrained_path)
        elif config_path:
            model_config = DALLEConfig.from_json_file(config_path)
            model = DALLEModel(model_config)
        else:
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