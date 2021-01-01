import torch
import numpy as np
import hshap
from hshap.src import Explainer


def f(batch):
    assert len(batch.size()) == 4 and batch.size(1) == 3
    predictions = np.array(
        [
            [0, 1] if torch.nonzero(el.sum(0), as_tuple=False).size(0) > 0 else [1, 0]
            for el in batch
        ]
    )
    return torch.from_numpy(predictions)


ref = torch.zeros((3, 9, 9))
hexp = Explainer(f, ref, M=4)
test_input = torch.zeros((3, 9, 9))
test_input[:, 0:3, 0:2] = 1
test_input[:, -1, 5:-1] = 1
test_input[:, 0, -1] = 1
saliency_map, _ = hexp.explain(test_input, label=1)
print(saliency_map)

# DEFINE PERTURBATION SIZES
exp_x = np.linspace(-1, 0, 20)
perturbation_sizes = np.sort(1.1 - 10 ** (exp_x))
perturbations_L = len(perturbation_sizes)

activation_threshold = 0
salient_points = np.where(saliency_map > activation_threshold)
salient_rows = salient_points[0]
salient_columns = salient_points[1]
scores = saliency_map[salient_points]
L = len(scores)
ranks = np.argsort(scores)
print(ranks)

for k, perturbation_size in enumerate(perturbation_sizes):
    perturbed_img = test_input.clone()
    perturbation_L = round(L * perturbation_size)
    perturbed_ids = ranks[-perturbation_L:]
    print(perturbed_ids)
    perturbed_rows = salient_rows[perturbed_ids]
    pertubed_columns = salient_columns[perturbed_ids]
    perturbed_img[:, perturbed_rows, pertubed_columns] = ref[
        :, perturbed_rows, pertubed_columns
    ]