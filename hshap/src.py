import numpy as np
from scipy.special import comb
from itertools import permutations
from functools import reduce
import torch


class Node:
    def __init__(self, depth=0, path=None, score=None):
        self.depth = depth
        self.path = path
        self.score = score

    def leaf(self, size, minW, minH):
        startRow, endRow, startColumn, endColumn = self.pathMaskCoordinates(size)
        rootw = endColumn - startColumn
        rooth = endRow - startRow
        if (rootw < 2 * minW) or (rooth < 2 * minH):
            return True
        else:
            return False

    def __mask2str(self, mask):
        return reduce(lambda a, b: str(a) + str(b), mask.astype(int))

    def __str2mask(self, string):
        L = len(string)
        mask = np.empty((L,))
        for i in range(L):
            mask[i] = int(string[i])
        return mask

    def pathMaskCoordinates(self, size):
        path = self.path
        startRow = 0
        startColumn = 0
        endRow = size[0]
        endColumn = size[1]
        if path is not None:
            for layer in path:
                w = endColumn - startColumn
                h = endRow - startRow
                feature_index = np.where(layer == 1)[0]
                if feature_index == 0:
                    endRow = startRow + h / 2
                    endColumn = startColumn + w / 2
                elif feature_index == 1:
                    endRow = startRow + h / 2
                    startColumn += w / 2
                elif feature_index == 2:
                    startRow += h / 2
                    endColumn = startColumn + w / 2
                elif feature_index == 3:
                    startRow += h / 2
                    startColumn += w / 2
        return (round(startRow), round(endRow), round(startColumn), round(endColumn))

    def computeShap(self, masks, feature, predictions_dictionary):
        feature_index = np.where(feature == 1)
        subset_indices = np.where(masks[:, feature_index[0][0]] == 0)
        subset = masks[subset_indices]
        added_subset = [np.add(sub, feature) for sub in subset]
        diffs = [
            1
            / comb(len(feature) - 1, np.sum(b))
            * (
                predictions_dictionary[self.__mask2str(a)]
                - predictions_dictionary[self.__mask2str(b)]
            )
            for a, b in zip(added_subset, subset)
        ]
        phi = np.sum(diffs) / len(feature)
        return phi

    def masked_inputs(self, masks, _input, background):
        start_row, end_row, start_column, end_column = self.pathMaskCoordinates(
            background[0].shape
        )
        root_input = background.clone()
        root_input[:, start_row:end_row, start_column:end_column] = _input[
            :, start_row:end_row, start_column:end_column
        ]
        d = len(root_input.shape)
        q = list(np.ones(d + 1, dtype=np.integer))
        q[0] = len(masks)
        masked_inputs = root_input.repeat(q)
        w = end_column - start_column
        h = end_row - start_row
        for i, mask in enumerate(masks):
            maskIndices = np.where(mask == 0)[0]
            for index in maskIndices:
                maskStartRow = start_row
                maskEndRow = end_row
                maskStartColumn = start_column
                maskEndColumn = end_column
                # First quadrant
                if index == 0:
                    maskEndRow = start_row + h / 2
                    maskEndColumn = start_column + w / 2
                # Second quadrant
                elif index == 1:
                    maskEndRow = start_row + h / 2
                    maskStartColumn += w / 2
                # Third quadrant
                elif index == 2:
                    maskStartRow += h / 2
                    maskEndColumn = start_column + w / 2
                # Fourth quadrant
                elif index == 3:
                    maskStartRow += h / 2
                    maskStartColumn += w / 2
                maskStartRow = round(maskStartRow)
                maskEndRow = round(maskEndRow)
                maskStartColumn = round(maskStartColumn)
                maskEndColumn = round(maskEndColumn)
                masked_inputs[
                    i, :, maskStartRow:maskEndRow, maskStartColumn:maskEndColumn
                ] = background[
                    :, maskStartRow:maskEndRow, maskStartColumn:maskEndColumn
                ]
        return masked_inputs

    def children_scores(self, masks, features, outputs):
        if self.leaf is True:
            return []
        else:
            outputs_dictionary = {
                self.__mask2str(mask): outputs[i] for i, mask in enumerate(masks)
            }
            return [
                self.computeShap(masks, feature, outputs_dictionary)
                for feature in features
            ]

    def child(self, feature, score):
        if self.leaf is True:
            return None
        else:
            return Node(
                depth=self.depth + 1,
                path=np.concatenate((self.path, np.array([self.__str2mask(feature)])))
                if self.path is not None
                else np.array([self.__str2mask(feature)]),
                score=score,
            )


class Explainer:
    def __init__(self, model, background, M=4):
        self.input = None
        self.label = None
        self.minW = None
        self.minH = None
        self.model = model
        self.background = background
        self.size = (self.background.shape[1], self.background.shape[2])
        self.M = M
        self.masks = self.generateMasks()
        self.features = np.identity(self.M, dtype=np.bool).reshape((self.M, self.M))
        print(r"Initialized explainer with map size {}".format(self.size))

    def generateMasks(self):
        masks = np.ones((1, self.M), dtype=np.bool)
        for i in range(self.M):
            s = np.zeros(self.M, dtype=np.bool)
            s[0:i] = 1
            p = permutations(s)
            a = np.array(list(set(p)))
            masks = np.concatenate((masks, a))
        return masks

    def flatten(self, l):
        for el in l:
            if isinstance(el, list) and not isinstance(el, (str, bytes)):
                yield from self.flatten(el)
            else:
                yield el

    def addNodeMask(self, node, map):
        startRow, endRow, startColumn, endColumn = node.pathMaskCoordinates(self.size)
        print(startRow, endRow, startColumn, endColumn, node.score)
        nodeArea = (endRow - startRow) * (endColumn - startColumn)
        map[startRow:endRow, startColumn:endColumn] = node.score
        # map[startRow : endRow, startColumn : endColumn] = node.score / nodeArea

    def explain(
        self,
        _input,
        label,
        minW=2,
        minH=2,
        threshold_mode="absolute",
        threshold=0,
        percentile=50,
    ):
        self.input = _input
        self.label = label
        self.minW = minW
        self.minH = minH
        batch_size = 2
        main_node = Node()
        level = [main_node]
        leafs = []
        L = len(level)
        while L > 0:
            layer_scores = np.zeros((L, self.M))
            n_batch = (
                int(L / batch_size) if L % batch_size == 0 else int(L / batch_size) + 1
            )
            for batch_id in np.arange(0, n_batch):
                first_id = batch_size * batch_id
                if batch_id < n_batch:
                    batch = level[first_id : first_id + batch_size]
                else:
                    batch = level[first_id:]
                l = len(batch)
                with torch.no_grad():
                    batch_input = torch.cat(
                        [
                            node.masked_inputs(self.masks, _input, self.background)
                            for node in batch
                        ],
                        0,
                    )
                    batch_outputs = self.model(batch_input).cpu()
                    batch_outputs = batch_outputs.view(
                        (l, 2 ** self.M, batch_outputs.size(1))
                    )
                    for i, node in enumerate(batch):
                        node_outputs = batch_outputs[i]
                        # _, node_outputs = torch.max(node_outputs, 1)
                        node_outputs = node_outputs[:, label]
                        node_scores = node.children_scores(
                            self.masks, self.features, node_outputs
                        )
                        layer_scores[batch_id * batch_size + i] = node_scores
                    torch.cuda.empty_cache()
            flat_layer_scores = layer_scores.flatten()
            # print(flat_layer_scores)
            if threshold_mode == "absolute":
                masked_layer_scores = np.ma.masked_greater(
                    flat_layer_scores, threshold
                ).mask.reshape(layer_scores.shape)
            if threshold_mode == "relative":
                threshold = np.percentile(flat_layer_scores, percentile)
                if threshold <= 0:
                    masked_layer_scores = np.ma.masked_greater(
                        flat_layer_scores, threshold
                    ).mask.reshape(layer_scores.shape)
                else:
                    masked_layer_scores = np.ma.masked_greater_equal(
                        flat_layer_scores, threshold
                    ).mask.reshape(layer_scores.shape)
            # print(percentile, threshold, max(flat_layer_scores))
            next_level = []
            for i, node in enumerate(level):
                for j, relevant in enumerate(masked_layer_scores[i]):
                    if relevant == True:
                        child_score = layer_scores[i, j]
                        feature = self.features[j]
                        child = node.child(feature, child_score)
                        if child.leaf(self.size, self.minW, self.minH) is True:
                            leafs.append(child)
                        else:
                            next_level.append(child)
            level = next_level
            L = len(level)
            # print(L, len(leafs))
        saliency_map = np.zeros(self.size)
        for node in leafs:
            self.addNodeMask(node, saliency_map)
        return saliency_map, leafs
