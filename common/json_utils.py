import os
import json


def load_from_json(file_path, message=None, print_msg=True):
    _file_path = os.path.expanduser(file_path)

    f = open(_file_path, "r")
    result = json.load(f)

    if message is None:
        message = f"Loaded Data from '{_file_path}'"

    if print_msg:
        print(message)

    return result


def dump_to_json(file_path, data, message=None):
    _file_path = os.path.expanduser(file_path)

    dir_name = os.path.dirname(_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    f = open(_file_path, "w")
    json.dump(data, f, indent=4, default=to_jsonable)

    if message is None:
        message = f"Saved Data to '{_file_path}'"
    print(message)


def to_jsonable(obj):
    if isinstance(obj, set) or isinstance(obj, tuple):
        return list(obj)
    else:
        return str(obj)


def pretty_print_json(data):
    """
    prints given data in pretty format
    """
    print(json.dumps(data, indent=4, default=to_jsonable))
