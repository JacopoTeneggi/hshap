from typing import Generator, Iterable, Tuple
import torch
from torch import Tensor
from itertools import permutations
import numpy as np
from functools import reduce
import time

factorial = np.math.factorial


def enumerate_batches(
    collection: Iterable, batch_size: int
) -> Generator[Tuple[int, list], None, None]:
    """
    Batch enumerator
    """
    L = len(collection)
    for i, first_el in enumerate(range(0, L, batch_size)):
        last_el = first_el + batch_size
        if last_el < L:
            yield i, collection[first_el:last_el]
        else:
            yield i, collection[first_el:]


def hshap_features(M: int) -> np.ndarray:
    """
    Make the required M features
    """
    return np.identity(M, dtype=int).reshape((M, M))


def make_masks(M: int) -> np.ndarray:
    """
    Make all required masks to compute Shapley values given the number of features M
    """
    masks = np.ones((1, M), dtype=bool)
    for i in range(M):
        s = np.zeros(M, dtype=bool)
        s[0:i] = 1
        p = permutations(s)
        a = np.array(list(set(p)))
        masks = np.concatenate((masks, a))
    return masks


def mask(path: np.ndarray, x: Tensor, _x: Tensor) -> torch.Tensor:
    """
    Creates a masked copy of x based on node.path and the specified background
    """
    if sum(path[-1]) == 0:
        return _x
    else:
        coords = np.array([[0, 0], [_x.size(1), _x.size(2)]], dtype=int)
        for level in path[1:-1]:
            if sum(level) == 1:
                center = (
                    (coords[0][0] + coords[1][0]) / 2,
                    (coords[0][1] + coords[1][1]) / 2,
                )
                feature_id = np.where(level == 1)[0]
                (feature_row, feature_column) = (int(feature_id / 2), feature_id % 2)
                coords[0][0] += feature_row * center[0]
                coords[0][1] += feature_column * center[1]
                coords[1][0] -= (1 - feature_row) * center[0]
                coords[1][1] -= (1 - feature_column) * center[1]
        level = path[-1]
        center = ((coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2)
        feature_ids = np.where(level == 1)[0]
        for feature_id in feature_ids:
            (feature_row, feature_column) = (int(feature_id / 2), feature_id % 2)
            feature_coords = coords.copy()
            feature_coords[0][0] += feature_row * center[0]
            feature_coords[0][1] += feature_column * center[1]
            feature_coords[1][0] -= (1 - feature_row) * center[0]
            feature_coords[1][1] -= (1 - feature_column) * center[1]
            _x[
                :,
                feature_coords[0][0] : feature_coords[1][0],
                feature_coords[0][1] : feature_coords[1][1],
            ] = x[
                :,
                feature_coords[0][0] : feature_coords[1][0],
                feature_coords[0][1] : feature_coords[1][1],
            ]
        return _x


def mask2str(mask: np.ndarray) -> str:
    """
    Convert a mask from array to string
    """
    return reduce(lambda a, b: str(a) + str(b), mask.astype(int))


def str2mask(string: str) -> np.ndarray:
    """
    Convert a string into mask
    """
    L = len(string)
    mask = np.empty((L,))
    for i in range(L):
        mask[i] = int(string[i])
    return mask


DEFAULT_M = 4
DEFAULT_MASKS = make_masks(DEFAULT_M)
DEFAULT_FEATURES = hshap_features(DEFAULT_M)


def shapley_phi(
    logits_dictionary: dict, feature: np.ndarray, masks: np.ndarray = DEFAULT_MASKS
) -> float:
    """
    Compute Shapley coefficient of a feature
    """
    d = len(feature)
    feature_id = np.where(feature == 1)
    _set_id = np.where(masks[:, feature_id[0][0]] == 0)
    _set = masks[_set_id]
    phi = 0
    for s in _set:
        sUi = s + feature
        phi += (
            factorial(sum(s))
            * factorial(d - sum(s) - 1)
            / factorial(d)
            * (logits_dictionary[mask2str(sUi)] - logits_dictionary[mask2str(s)])
        )
    return phi


def children_scores(
    label_logits: Tensor,
    masks: np.ndarray = DEFAULT_MASKS,
    features: np.ndarray = DEFAULT_FEATURES,
) -> np.ndarray:
    """
    Compute Shapley coefficients of children features
    """
    logits_dictionary = {
        mask2str(mask): label_logits[i] for i, mask in enumerate(masks)
    }
    return np.array(
        [shapley_phi(logits_dictionary, feature, masks) for feature in features],
        dtype=object,
    )


def compute_perturbed_logits(model, image, explanation, perturbation_sizes):
    # DEFINE PERTURBATION SIZES
    # exp_x = np.linspace(-1, 0, 20)
    # perturbation_sizes = np.sort(1.1 - 10 ** (exp_x))
    # perturbations_L = len(perturbation_sizes)

    # IDENTIFY SALIENT POINTS AND RANK THEM
    activation_threshold = 0
    salient_points = np.where(explanation > activation_threshold)
    salient_rows = salient_points[0]
    salient_columns = salient_points[1]
    scores = explanation[salient_points]
    L = len(scores)
    ranks = np.argsort(scores)

    # PERTURBATE IMAGES AND EVALUATE LOGITS
    perturbed_batch = image.unsqueeze(0).repeat(perturbations_L, 1, 1, 1)
    for k, perturbation_size in enumerate(perturbation_sizes):
        print("Perturbation={}".format(perturbation_size))

        perturbation_L = round(perturbation_size * L)
        perturbed_ids = ranks[-perturbation_L:]
        perturbed_rows = salient_rows[perturbed_ids]
        perturbed_columns = salient_columns[perturbed_ids]
        perturbed_batch[k, :, perturbed_rows, perturbed_columns] = ref[
            :, perturbed_rows, perturbed_columns
        ]

    with torch.no_grad():
        outputs = model(perturbed_batch)
        del perturbed_batch
        torch.cuda.empty_cache()
        logits = torch.log10(torch.nn.Softmax(dim=1)(outputs))

    return logits
