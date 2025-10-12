from common.yaml_utils import load_from_yaml


def load_secret(key: str, print_log=True):
    return load_from_yaml("./secrets.yaml", print_msg=print_log)[key]
