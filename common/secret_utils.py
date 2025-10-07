from common.yaml_utils import load_from_yaml


def load_secret(key: str) -> str:
    return load_from_yaml("./secrets.yaml")[key]
