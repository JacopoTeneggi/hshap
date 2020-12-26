import numpy as np
from scipy.special import comb
from itertools import permutations
from functools import reduce
import torch


class Node:
    def __init__(self, explainer, depth, path=None, score=None):
        self.explainer = explainer
        self.depth = depth
        self.path = path
        self.score = score
        self.root_coordinates = self.__pathMaskCoordinates()
        self.root_input = self.__rootPathInput()
        self.leaf = self.__leaf()

    def __rootPathInput(self):
        startRow, endRow, startColumn, endColumn = self.root_coordinates
        rootInput = self.explainer.background.clone()
        rootInput[:, startRow:endRow, startColumn:endColumn] = self.explainer.input[
            :, startRow:endRow, startColumn:endColumn
        ]
        return rootInput

    def __leaf(self):
        startRow, endRow, startColumn, endColumn = self.root_coordinates
        rootw = endColumn - startColumn
        rooth = endRow - startRow
        if (rootw < 2 * self.explainer.minW) or (rooth < 2 * self.explainer.minH):
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

    def __pathMaskCoordinates(self):
        path = self.path
        startRow = 0
        startColumn = 0
        endRow = self.explainer.h
        endColumn = self.explainer.w
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

    def computeShap(self, feature, predictions_dictionary):
        feature_index = np.where(feature == 1)
        subset_indices = np.where(self.explainer.masks[:, feature_index[0][0]] == 0)
        subset = self.explainer.masks[subset_indices]
        added_subset = [np.add(sub, feature) for sub in subset]
        diffs = [
            1
            / comb(self.explainer.M - 1, np.sum(b))
            * (
                predictions_dictionary[self.__mask2str(a)]
                - predictions_dictionary[self.__mask2str(b)]
            )
            for a, b in zip(added_subset, subset)
        ]
        phi = np.sum(diffs) / self.explainer.M
        return phi

    def masked_inputs(self):
        if self.leaf is True:
            return []
        else:
            startRow, endRow, startColumn, endColumn = self.root_coordinates
            d = len(self.root_input.shape)
            q = list(np.ones(d + 1, dtype=np.integer))
            q[0] = len(self.explainer.masks)
            masked_inputs = self.root_input.repeat(q)
            w = endColumn - startColumn
            h = endRow - startRow
            for i, mask in enumerate(self.explainer.masks):
                maskIndices = np.where(mask == 0)[0]
                for index in maskIndices:
                    maskStartRow = startRow
                    maskEndRow = endRow
                    maskStartColumn = startColumn
                    maskEndColumn = endColumn
                    # First quadrant
                    if index == 0:
                        maskEndRow = startRow + h / 2
                        maskEndColumn = startColumn + w / 2
                    # Second quadrant
                    elif index == 1:
                        maskEndRow = startRow + h / 2
                        maskStartColumn += w / 2
                    # Third quadrant
                    elif index == 2:
                        maskStartRow += h / 2
                        maskEndColumn = startColumn + w / 2
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
                    ] = self.explainer.background[
                        :, maskStartRow:maskEndRow, maskStartColumn:maskEndColumn
                    ]
            return masked_inputs

    def children(self, predictions):
        if self.leaf is True:
            return []
        else:
            predictions_dictionary = {
                self.__mask2str(mask): predictions[i]
                for i, mask in enumerate(self.explainer.masks)
            }
            children = [
                Node(
                    self.explainer,
                    self.depth + 1,
                    path=np.concatenate(
                        (self.path, np.array([self.__str2mask(feature)]))
                    )
                    if self.path is not None
                    else np.array([self.__str2mask(feature)]),
                    score=self.computeShap(feature, predictions_dictionary),
                )
                for feature in self.explainer.features
            ]
            return children


class Explainer:
    def __init__(self, model, background, M=4):
        self.input = None
        self.label = None
        self.minW = None
        self.minH = None
        self.model = model
        self.background = background
        self.h = self.background.shape[1]
        self.w = self.background.shape[2]
        self.M = M
        self.masks = self.generateMasks()
        self.features = np.identity(self.M, dtype=np.bool).reshape((self.M, self.M))
        print("Initialized explainer with map size (%d, %d)" % (self.h, self.w))

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
        startRow, endRow, startColumn, endColumn = node.root_coordinates
        nodeArea = (endRow - startRow) * (endColumn - startColumn)
        map[startRow:endRow, startColumn:endColumn] = node.score
        # map[startRow : endRow, startColumn : endColumn] = node.score / nodeArea

    def explain(
        self,
        input,
        label,
        minW=2,
        minH=2,
        threshold_mode="absolute",
        threshold=0,
        percentile=50,
    ):
        self.input = input
        self.label = label
        self.minW = minW
        self.minH = minH
        batch_size = 4
        main_node = Node(self, 0)
        level = [main_node]
        leafs = []
        L = len(level)
        while L > 0:
            next_level = []
            scores = []
            n_batch = (
                int(L / batch_size) if L % batch_size == 0 else int(L / batch_size) + 1
            )
            for batch_id in np.arange(n_batch):
                first_id = batch_size * batch_id
                if batch_id < n_batch:
                    batch = level[first_id : first_id + batch_size]
                else:
                    batch = level[first_id:]
                l = len(batch)
                with torch.no_grad():
                    batch_input = torch.cat([node.masked_inputs() for node in level], 0)
                    batch_outputs = self.model(batch_input)
                    batch_outputs = batch_outputs.view(
                        (l, 2 ** self.M, batch_outputs.size(1))
                    )
                    for i, node in enumerate(batch):
                        node_outputs = batch_outputs[i]
                        _, node_predictions = torch.max(node_outputs, 1)
                        node_children = node.children(node_predictions)
                        for child in node_children:
                            next_level.append(child)
                            scores.append(child.score)
            scores = np.array(scores)
            next_level = np.array(next_level)
            if threshold_mode == "absolute":
                relevant_children = np.where(scores > threshold)[0]
                next_level = next_level[relevant_children]
            if threshold_mode == "relative":
                threshold = np.percentile(scores, percentile)
                relevant_children = np.where(scores >= threshold)[0]
                next_level = next_level[relevant_children]
            leaf_ids = []
            for i, node in enumerate(next_level):
                if node.leaf is True:
                    leafs.append(node)
                    leaf_ids.append(i)
            level = np.delete(next_level, leaf_ids)
            L = len(level)
        saliency_map = np.zeros((self.h, self.w))
        for node in leafs:
            self.addNodeMask(node, saliency_map)
        return saliency_map, leafs
