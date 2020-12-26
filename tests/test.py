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


hexp = Explainer(f, torch.zeros((3, 9, 9)), M=4)
test_input = torch.zeros((3, 9, 9))
test_input[:, 0:3, 0:2] = 1
saliency_map, _ = hexp.explain(test_input, label=1)
print(saliency_map)
