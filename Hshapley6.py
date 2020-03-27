import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class HierarchicalShap:
    """
    Explains the salient region of images for a given network.
    """
    def __init__(self, model, mean=np.array([0.5, 0.5, 0.5]), sd=np.array([0.5, 0.5, 0.5]), background_type = "white",  background = None):
        """
        Parameters
        ----------
        model: the model from which you wish to study the decision
        mean: the mean used for image normlization
        sd: the standard deviation used for normalization
        background_type: pre-defined background images. Can chose from {"white", "black"}
        background: if you want  to use a specified background
        """
        self.model = model
        self.mean = mean
        self.sd = sd
        self.background = background
        self.background_type = background_type

    def display_cropped_images(self, images, score):
        mean = np.array([0.5, 0.5, 0.5])
        sd = np.array([0.5, 0.5, 0.5])
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        for i in range(4):
            for j in range(4):
                sample_image = images[4 * i + j].numpy().transpose(1, 2, 0)
                im = sample_image * sd + mean
                axs[i, j].imshow(im)
                axs[i, j].set_title("#%d score:%f " % (4 * i + j, score[4 * i + j]))

    def construct_subsets(self, im, start=(0, 0), region_size=(None, None)):
        if (region_size[0] == None or region_size[1] == None):
            start = (0, 0)
            region_size = im.numpy().shape[1:3]

        middle = (start[0] + region_size[0]//2, start[1] + region_size[1]//2)
        end = (start[0] + region_size[0], start[1] + region_size[1])


        top_left = (start, (middle[0] - start[0], middle[1] - start[1]))
        top_right = ((start[0], middle[1]), (middle[0]-start[0], end[1]-middle[1]))
        bottom_left = ((middle[0], start[1]), (end[0]-middle[0], middle[1]-start[1]))
        bottom_right = (middle, (end[0]-middle[0], end[1]-middle[1]))
        regions = np.array([[top_left, top_right],[bottom_left, bottom_right]])


        subsets_size = [16]
        image_size = []
        for dim in im.shape:
            subsets_size.append(dim)
            image_size.append(dim)
        subsets = torch.zeros(subsets_size)


        if (self.background == None):
            if (self.background_type == "white"):
                bg = torch.ones(image_size)
            else:
                NameError("Unknown background type. Choose white or give a specified background.")
        else:
            print(self.background)
            figure(), imshow(self.background)
            bg = self.background[:, start[0]:start[0] + region_size[0], start[1]:start[1] + region_size[1]]
            print(bg)
            figure(), imshow(bg)

        # removing 0 features
        im1234 = bg.clone()
        im1234[:, start[0]:end[0], start[1]:end[1]] = im[:, start[0]:end[0], start[1]:end[1]]
        # removing 1 feature
        im234 = im1234.clone()
        im234[:, start[0]:middle[0], start[1]:middle[1]] = bg[:, start[0]:middle[0], start[1]:middle[1]]
        im134 = im1234.clone()
        im134[:, start[0]:middle[0], middle[1]:end[1]] = bg[:, start[0]:middle[0], middle[1]:end[1]]
        im124 = im1234.clone()
        im124[:, middle[0]:end[0], start[1]:middle[1]] = bg[:, middle[0]:end[0], start[1]:middle[1]]
        im123 = im1234.clone()
        im123[:, middle[0]:end[0], middle[1]:end[1]] = bg[:, middle[0]:end[0], middle[1]:end[1]]
        # removing 2 features
        im34 = im234.clone()
        im34[:, start[0]:middle[0], middle[1]:end[1]] = bg[:, start[0]:middle[0], middle[1]:end[1]]
        im24 = im234.clone()
        im24[:, middle[0]:end[0], start[1]:middle[1]] = bg[:, middle[0]:end[0], start[1]:middle[1]]
        im23 = im234.clone()
        im23[:, middle[0]:end[0], middle[1]:end[1]] = bg[:, middle[0]:end[0], middle[1]:end[1]]
        im14 = im134.clone()
        im14[:, middle[0]:end[0], start[1]:middle[1]] = bg[:, middle[0]:end[0], start[1]:middle[1]]
        im13 = im134.clone()
        im13[:, middle[0]:end[0], middle[1]:end[1]] = bg[:, middle[0]:end[0], middle[1]:end[1]]
        im12 = im123.clone()
        im12[:, middle[0]:end[0], start[1]:middle[1]] = bg[:, middle[0]:end[0], start[1]:middle[1]]
        # removing 3 features
        im4 = im34.clone()
        im4[:, middle[0]:end[0], start[1]:middle[1]] = bg[:, middle[0]:end[0], start[1]:middle[1]]
        im3 = im34.clone()
        im3[:, middle[0]:end[0], middle[1]:end[1]] = bg[:, middle[0]:end[0], middle[1]:end[1]]
        im2 = im24.clone()
        im2[:, middle[0]:end[0], middle[1]:end[1]] = bg[:, middle[0]:end[0], middle[1]:end[1]]
        im1 = im14.clone()
        im1[:, middle[0]:end[0], middle[1]:end[1]] = bg[:, middle[0]:end[0], middle[1]:end[1]]
        # removing 4
        im_ = bg

        subsets[0] = im1234
        subsets[1] = im234
        subsets[2] = im134
        subsets[3] = im124
        subsets[4] = im123
        subsets[5] = im34
        subsets[6] = im24
        subsets[7] = im23
        subsets[8] = im14
        subsets[9] = im13
        subsets[10] = im12
        subsets[11] = im4
        subsets[12] = im3
        subsets[13] = im2
        subsets[14] = im1
        subsets[15] = im_

        return subsets, regions

    def subsetScores(self, sub, label):
        outputs = self.model(sub)

        score = np.zeros(16)
        score[0] = outputs[0, label]
        score[1] = outputs[1, label]
        score[2] = outputs[2, label]
        score[3] = outputs[3, label]
        score[4] = outputs[4, label]
        score[5] = outputs[5, label]
        score[6] = outputs[6, label]
        score[7] = outputs[7, label]
        score[8] = outputs[8, label]
        score[9] = outputs[9, label]
        score[10] = outputs[10, label]
        score[11] = outputs[11, label]
        score[12] = outputs[12, label]
        score[13] = outputs[13, label]
        score[14] = outputs[14, label]
        score[15] = outputs[15, label]
        return score

    def constructShapMap(self, score):
        phi1 = (score[14] - score[15] + score[0] - score[1]) / 4 \
               + (score[8] - score[11] + score[9] - score[12] + score[10] - score[13]
                  + score[2] - score[5] + score[3] - score[6] + score[4] - score[7]) / 12
        # verified

        phi2 = (score[13] - score[15] + score[0] - score[2]) / 4 \
               + (score[6] - score[11] + score[7] - score[12] + score[10] - score[14]
                  + score[1] - score[5] + score[3] - score[8] + score[4] - score[9]) / 12
        # verified


        phi3 = (score[12] - score[15] + score[0] - score[3]) / 4 \
               + (score[5] - score[11] + score[9] - score[14] + score[7] - score[13]
                  + score[2] - score[8] + score[1] - score[6] + score[4] - score[10]) / 12
        # verified

        phi4 = (score[11] - score[15] + score[0] - score[4]) / 4 \
               + (score[8] - score[14] + score[5] - score[12] + score[6] - score[13]
                  + score[1] - score[7] + score[3] - score[10] + score[2] - score[9]) / 12
        # verified

        shap_map = np.array([[phi1, phi2], [phi3, phi4]])
        return shap_map

    def get_salient_regions(self, shap_map, shapTol, regions):
        srs = []
        for i in range(len(shap_map)):
            for j in range(len(shap_map)):
                if (shap_map[i, j] > shapTol):
                    srs.append(regions[i,j])
        return srs

    def display_salient(self, im, srs_coll):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        sample_image = im.numpy().transpose(1, 2, 0)
        image = sample_image * self.sd + self.mean
        n_points = 101
        ax1.imshow(image)
        ax2.imshow(image)
        ax3.imshow(image)
        for srs in srs_coll:
            for sr in srs:
                start = sr[0]
                quadrant_size = sr[1]
                ax2.plot(np.linspace(start[1], start[1] + quadrant_size[1], n_points), start[0] * np.ones(n_points),
                         'r')
                ax2.plot(np.linspace(start[1], start[1] + quadrant_size[1], n_points),
                         (start[0] + quadrant_size[0]) * np.ones(n_points), 'r')
                ax2.plot(start[1] * np.ones(n_points), np.linspace(start[0], start[0] + quadrant_size[0], n_points),
                         'r')
                ax2.plot((start[1] + quadrant_size[1]) * np.ones(n_points),
                         np.linspace(start[0], start[0] + quadrant_size[0], n_points), 'r')
                xs = [start[1], start[1] + quadrant_size[1], start[1] + quadrant_size[1], start[1]]
                ys = [start[0], start[0], start[0] + quadrant_size[0], start[0] + quadrant_size[0]]
                ax3.fill(xs, ys, 'r', alpha=1 / len(srs_coll))
        ax1.set_xlim([0, im.shape[2]])
        ax1.set_ylim([im.shape[1], 0])
        ax2.set_xlim([0, im.shape[2]])
        ax2.set_ylim([im.shape[1], 0])
        ax3.set_xlim([0, im.shape[2]])
        ax3.set_ylim([im.shape[1], 0])

    def do_all(self, im, label, strt, region_size, shapTol, debug=False):

        images_final, regions = self.construct_subsets(im, strt, region_size)
        score = self.subsetScores(images_final, label)
        sm = self.constructShapMap(score)
        if (debug):
            self.display_cropped_images(images_final, score)
            f = plt.figure()
            sns.heatmap(sm)
            f.suptitle("Shap values of each quadrant");

        srs = self.get_salient_regions(sm, shapTol, regions)

        return srs

    def shapMap(self, image, label, shapTol=[6], keepItSimple=False, debug=False):
        ls = []
        delta = [image.shape[1]//20, image.shape[2]//24]
        xf = [image.shape[1], image.shape[2]]
        starts = [(0, 0), (0, delta[1]), (0, 2 * delta[1]), (delta[0], 0), (2 * delta[0], 0), (delta[0], 2 * delta[1]),
                  (2 * delta[0], delta[1]), (delta[0], delta[1]), (2 * delta[0], 2 * delta[1])]
        ends = [(xf[0], xf[1]), (xf[0], xf[1] - delta[1]), (xf[0], xf[1] - 2 * delta[1]), (xf[0] - delta[0], xf[1]),
                (xf[0] - 2 * delta[0], xf[1]), (xf[0] - delta[0], xf[1] - 2 * delta[1]),
                (xf[0] - 2 * delta[0], xf[1] - delta[1]), (xf[0] - delta[0], xf[1] - delta[1]),
                (xf[0] - 2 * delta[0], xf[1] - 2 * delta[1])]
        if (keepItSimple):
            starts = [(0, 0)]
            ends = [(100, 120)]
        for start in starts:
            for end in ends:
                size = (end[0]-start[0], end[1]-start[1])
                for tol in shapTol:
                    srs = [(start, size)]
                    finished = []
                    k = 0
                    while (len(srs) > 0):
                        all = []
                        for sr in srs:
                            s = self.do_all(image, label, sr[0], sr[1], tol, debug)
                            if (s == []):
                                finished.append(((sr[0]), (sr[1])))
                            else:
                                all += s
                        srs = all
                        k += 1
                    ls.append(finished)
        self.display_salient(image, ls)
