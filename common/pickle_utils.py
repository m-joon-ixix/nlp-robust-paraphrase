import os
import pickle


def load_from_pickle(file_path, message=None, print_msg=True):
    _file_path = os.path.expanduser(file_path)

    if message is None:
        message = f"Loading Data from '{_file_path}'"

    if print_msg:
        print(message)

    f = open(_file_path, "rb")
    result = pickle.load(f)

    return result


def dump_to_pickle(file_path, data, message=None):
    _file_path = os.path.expanduser(file_path)

    if message is None:
        message = f"Saving Data to '{_file_path}'"

    print(message)

    dir_name = os.path.dirname(_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    f = open(_file_path, "wb")
    pickle.dump(data, f)
