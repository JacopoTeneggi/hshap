import numpy as np
from scipy.special import comb
from itertools import permutations
from functools import reduce


class Node:
    """
    Represents the single feature
    """

    def __init__(self, explainer, depth, M, features, masks, path=None, score=None):
        self.explainer = explainer
        self.depth = depth
        self.M = M
        self.features = features
        self.masks = masks
        self.path = path
        self.score = score

    def computeShap(self, feature, predictions):
        feature_index = np.where(feature == 1)
        subset_indices = np.where(self.masks[:, feature_index[0][0]] == 0)
        subset = self.masks[subset_indices]
        added_subset = [np.add(sub, feature) for sub in subset]
        deltas = np.array(list(zip(added_subset, subset)))
        diffs = [1/comb(self.M - 1, np.sum(b)) * (predictions[self.mask2str(a)
                                                              ] - predictions[self.mask2str(b)]) for a, b in deltas]
        phi = np.sum(diffs) / self.M
        return phi.detach().numpy()

    def mask2path(self, mask):
        if self.path is None:
            return [mask]
        else:
            return np.concatenate((self.path, mask))

    def mask2str(self, mask):
        return reduce(lambda a, b: str(a) + str(b), mask.astype(int))

    def str2mask(self, string):
        L = len(string)
        mask = np.empty((L,))
        for i in range(L):
            mask[i] = int(string[i])
        return mask

    def pathMaskCoordinates(self, path, startRow, endRow, startColumn, endColumn):
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
        return int(startRow), int(endRow), int(startColumn), int(endColumn)

    def maskInput(self, mask, rootInput, startRow, endRow, startColumn, endColumn):
        w = endColumn - startColumn
        h = endRow - startRow
        if (type(rootInput) == np.ndarray):
            maskedInput = rootInput.copy()
        else:
            maskedInput = rootInput.clone()
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
            maskStartRow = int(maskStartRow)
            maskEndRow = int(maskEndRow)
            maskStartColumn = int(maskStartColumn)
            maskEndColumn = int(maskEndColumn)
            maskedInput[:, maskStartRow:maskEndRow, maskStartColumn:maskEndColumn] = self.explainer.background[:,
                                                                                                maskStartRow:maskEndRow, maskStartColumn:maskEndColumn]
        if (type(maskedInput) is not np.ndarray):
            maskedInput = maskedInput.view(-1, 3, self.explainer.h, self.explainer.w)
        return maskedInput

    def rootPathInput(self, path, input, background):
        startRow, endRow, startColumn, endColumn = self.pathMaskCoordinates(
            path, 0, self.explainer.h, 0, self.explainer.w)
        if (type(background) == np.ndarray):
            rootInput = background.copy()
        else:
            rootInput = background.clone()
        rootInput[:, startRow:endRow+1, startColumn:endColumn +
                  1] = input[:, startRow:endRow+1, startColumn:endColumn+1]
        return rootInput, startRow, endRow, startColumn, endColumn

    def nodeScores(self, input, label, threshold, minW, minH):
        #
        rootInput, startRow, endRow, startColumn, endColumn = self.rootPathInput(
            self.path, input, self.explainer.background)
        rootw = endColumn - startColumn
        rooth = endRow - startRow
        # Stop when it reaches the deepest layer and return current node
        if (rootw < 2*minW) or (rooth < 2*minH):
            return self
        # If not, go down another level and compute shap coefficients for features
        predictions = {self.mask2str(mask): self.explainer.model(self.maskInput(
            mask, rootInput, startRow, endRow, startColumn, endColumn))[:, label] for mask in self.masks}
        phis = {self.mask2str(feature): self.computeShap(
            feature, predictions) for feature in self.features}

        # Update number of computed features
        self.explainer.computed += self.M

        # Convert SHAP dictionary to lists -> TODO: evaluate wether SHAP dictionary is necessary
        values = np.fromiter(phis.values(), dtype=float)
        keys = list(phis.keys())

        # Identify relevant features
        if threshold is not None:
            relevantIndices = np.where(values > threshold)[0]
        else:
            relevantIndices = np.arange(self.M)

        # Update number of rejected features
        self.explainer.rejected += self.M - len(relevantIndices)

        # Initialize children and recursively compute SHAP values
        children = []
        for relevantIndex in relevantIndices:
            childPath = np.array([self.str2mask(keys[relevantIndex])])
            if self.path is not None:
                childPath = np.concatenate((self.path, childPath))
            child = Node(self.explainer, self.depth + 1, self.M,
                         self.features, self.masks, path=childPath, score=values[relevantIndex])
            children.append(child.nodeScores(
                input, label, threshold, minW, minH))
        return children


class Explainer:

    def __init__(self, model, background, M=4):
        self.model = model
        self.computed = None
        self.rejected = None
        self.background = background
        self.h = self.background.shape[1]
        self.w = self.background.shape[2]
        self.M = M
        self.masks = self.generateMasks()
        self.features = np.identity(
            self.M, dtype=np.bool).reshape((self.M, self.M))

    def generateMasks(self):
        # initialize masks array with all features on -> no need to compute permutations fro |S| = M
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
        startRow, endRow, startColumn, endColumn = node.pathMaskCoordinates(
            node.path, 0, self.h, 0, self.w)
        nodeArea = (endRow + 1 - startRow) * (endColumn + 1 - startColumn)
        map[startRow:endRow+1, startColumn:endColumn+1] = node.score / nodeArea

    def explain(self, input, label=None, threshold=0, minW=2, minH=2):
        self.computed = 0
        self.rejected = 0
        mainNode = Node(
            self, 0, 4, self.features, self.masks, score=1)
        nodes = mainNode.nodeScores(
            input, label, threshold, minW, minH)
        flatnodes = list(self.flatten(nodes))
        saliency_map = np.zeros((self.h, self.w))
        for node in flatnodes:
            self.addNodeMask(node, saliency_map)
        return saliency_map, flatnodes
