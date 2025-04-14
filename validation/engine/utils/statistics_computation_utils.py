import copy

import torch
from pytod.models.knn import KNN


def filter_outliers(input_data: torch.Tensor) -> torch.Tensor:
    """Function for filtering the outliers"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = KNN(device=device.type)

    # normalizing clip scores to range [0, 1] and sorting
    input_sorted, _ = torch.sort(input_data)
    input_sorted = input_sorted.reshape(-1, 1)

    # searching for the anomalies
    clf.fit(input_sorted)
    preds = torch.tensor(clf.labels_).to(device)

    outliers = torch.nonzero(preds == 1, as_tuple=True)[0]

    input_filtered = copy.deepcopy(input_sorted)
    input_filtered[outliers] = -1

    return input_filtered[input_filtered != -1]


def compute_mean(data: torch.Tensor, operation: str = "mean") -> torch.Tensor:
    """Function for computing average values for the input data using three different methods"""

    if operation == "mean":
        return data.mean()
    elif operation == "median":
        return data.median()
    elif operation == "geometric_mean":
        return torch.exp(torch.log(data).mean())
    else:
        raise ValueError(f"Unknown operation [{operation}] was specified.")
