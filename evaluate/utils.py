import numpy as np
from typing import List


def compute_accuracy(data_list: List[dict], question_idxs=[0, 1, 2]) -> float:
    response_count = 0
    correct_count = 0

    for data in data_list:
        for q_idx in question_idxs:
            correct_list = data["responses_correct"][q_idx]  # list[bool]
            response_count += len(correct_list)
            correct_count += sum(correct_list)

    return correct_count / response_count


# XParaCon: cross(X)-PARAphrase CONsistency
def compute_xparacon(data_list: List[dict]) -> float:
    def accuracy(data: dict, q_idx: int) -> float:
        correct_list = data["responses_correct"][q_idx]
        return sum(correct_list) / len(correct_list)

    # NOTE: -log(avg(stddev(acc(q0), acc(q1), acc(q2))))
    stddev_list = []
    for data in data_list:
        accuracies = [accuracy(data, q_idx) for q_idx in range(3)]
        stddev_list.append(np.std(accuracies))

    return -np.log2(np.average(stddev_list))
