import yaml


def unscale(x, mmax, mmin):
    return x * (mmax - mmin) + mmin


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path: str):
    """load configuration file given configuration name

    Args:
        config_name (str): name of the configuration

    Returns:
        config: dictionray of configuration
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
