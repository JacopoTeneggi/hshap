import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class HierarchicalShap:
    """
    Explains the salient regions of images according a given network.
    """

    def __init__(self, model, background, mean=np.array([0.5, 0.5, 0.5]), sd=np.array([0.5, 0.5, 0.5])):
        """
        Parameters
        ----------
        model : the model from which you wish to study the decision
        background : used to remove the contribution of non-considered regions when constructing subsets

        mean : the mean used for image normalization (useful for plotting from input)
        sd : the standard deviation used for normalization (useful for plotting from input)
        """
        self.model = model
        self.background = background
        self.mean = mean
        self.sd = sd

    def display_cropped_images(self, images, scores):
        """
        Draw the subsets.
        Parameters
        ----------
        images : all the subsets to draw

        scores : the output score for a class 1
        """
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        for i in range(4):
            for j in range(4):
                im = images[4 * i + j].numpy().transpose(1, 2, 0)
                im = im * self.sd + self.mean
                axs[i, j].imshow(im)
                axs[i, j].set_title("#%d score:%f " % (4 * i + j, scores[4 * i + j]))

    def construct_subsets(self, im, s=(0, 0), region_size=(None, None)):
        """
        Construct the subsets of im: all possible image resulting from removing from im the content of 0, 1, 2, 3
        or all 4 quadrants of the region defined by start and region_size .
        Parameters
        ----------
        im : the image from which to extract subsets

        s : the top left pixel coordinates of the region analyzed, a tuple of

        region_size : the size of the region analyzed

        Returns
        --------
        subsets : the list of 16 images

        r_coord : a 2x2 array where each entry is a tuple of tuples; the first indicating the start
                  of the region and the second its size
        """

        m = (s[0] + region_size[0] // 2, s[1] + region_size[1] // 2)
        e = (s[0] + region_size[0], s[1] + region_size[1])

        top_left = (s, (m[0] - s[0], m[1] - s[1]))
        top_right = ((s[0], m[1]), (m[0] - s[0], e[1] - m[1]))
        bottom_left = ((m[0], s[1]), (e[0] - m[0], m[1] - s[1]))
        bottom_right = (m, (e[0] - m[0], e[1] - m[1]))
        r_coord = np.array([[top_left, top_right], [bottom_left, bottom_right]])

        subsets_size = [16, im.shape[0], im.shape[1], im.shape[2]]

        bg = self.background
        # removing 0 features
        im1234 = bg.clone()
        im1234[:, s[0]:e[0], s[1]:e[1]] = im[:, s[0]:e[0], s[1]:e[1]]
        # removing 1 feature
        im234 = im1234.clone()
        im234[:, s[0]:m[0], s[1]:m[1]] = bg[:, s[0]:m[0], s[1]:m[1]]
        im134 = im1234.clone()
        im134[:, s[0]:m[0], m[1]:e[1]] = bg[:, s[0]:m[0], m[1]:e[1]]
        im124 = im1234.clone()
        im124[:, m[0]:e[0], s[1]:m[1]] = bg[:, m[0]:e[0], s[1]:m[1]]
        im123 = im1234.clone()
        im123[:, m[0]:e[0], m[1]:e[1]] = bg[:, m[0]:e[0], m[1]:e[1]]
        # removing 2 features
        im34 = im234.clone()
        im34[:, s[0]:m[0], m[1]:e[1]] = bg[:, s[0]:m[0], m[1]:e[1]]
        im24 = im234.clone()
        im24[:, m[0]:e[0], s[1]:m[1]] = bg[:, m[0]:e[0], s[1]:m[1]]
        im23 = im234.clone()
        im23[:, m[0]:e[0], m[1]:e[1]] = bg[:, m[0]:e[0], m[1]:e[1]]
        im14 = im134.clone()
        im14[:, m[0]:e[0], s[1]:m[1]] = bg[:, m[0]:e[0], s[1]:m[1]]
        im13 = im134.clone()
        im13[:, m[0]:e[0], m[1]:e[1]] = bg[:, m[0]:e[0], m[1]:e[1]]
        im12 = im123.clone()
        im12[:, m[0]:e[0], s[1]:m[1]] = bg[:, m[0]:e[0], s[1]:m[1]]
        # removing 3 features
        im4 = im34.clone()
        im4[:, m[0]:e[0], s[1]:m[1]] = bg[:, m[0]:e[0], s[1]:m[1]]
        im3 = im34.clone()
        im3[:, m[0]:e[0], m[1]:e[1]] = bg[:, m[0]:e[0], m[1]:e[1]]
        im2 = im24.clone()
        im2[:, m[0]:e[0], m[1]:e[1]] = bg[:, m[0]:e[0], m[1]:e[1]]
        im1 = im14.clone()
        im1[:, m[0]:e[0], m[1]:e[1]] = bg[:, m[0]:e[0], m[1]:e[1]]
        # removing 4
        im_ = bg.clone()

        subsets = torch.zeros(size=subsets_size)
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

        return subsets, r_coord

    def subset_scores(self, sub, label):
        """
        Compute the scores of each subset input.
        Parameters
        ----------
        sub : the subsets of inputs

        label : the class label - typically 1 -  in which we're interested.

        Returns
        --------
        score : an array of the 16 scores for each input
        """
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

    def shapley_of_quadrants(self, score):
        """
        Return a 2x2 array which contains the Shapley values associated with each quadrant
        Parameters
        ----------
        score : the network evaluation for each subset

        Returns
        --------
        shapley_coefficients : an array of the 16 scores for each input
        """

        phi1 = (score[14] - score[15] + score[0] - score[1]) / 4 \
               + (score[8] - score[11] + score[9] - score[12] + score[10] - score[13]
                  + score[2] - score[5] + score[3] - score[6] + score[4] - score[7]) / 12

        phi2 = (score[13] - score[15] + score[0] - score[2]) / 4 \
               + (score[6] - score[11] + score[7] - score[12] + score[10] - score[14]
                  + score[1] - score[5] + score[3] - score[8] + score[4] - score[9]) / 12

        phi3 = (score[12] - score[15] + score[0] - score[3]) / 4 \
               + (score[5] - score[11] + score[9] - score[14] + score[7] - score[13]
                  + score[2] - score[8] + score[1] - score[6] + score[4] - score[10]) / 12

        phi4 = (score[11] - score[15] + score[0] - score[4]) / 4 \
               + (score[8] - score[14] + score[5] - score[12] + score[6] - score[13]
                  + score[1] - score[7] + score[3] - score[10] + score[2] - score[9]) / 12

        shapley_coefficients = np.array([[phi1, phi2], [phi3, phi4]])
        return shapley_coefficients

    def get_salient_regions(self, shapley_values, tol, regions):
        """
        Determine which of the 4 quadrants are salient, i.e. have Shapley value larger than tol
        Parameters
        ----------
        shapley_values : the Shapley coefficients associated with each quadrant

        tol : the specified tolerance for a sub-region to be considered salient

        regions : the coordinates associated with each quadrant

        Returns
        --------
        srs : a list of the coordinates of the quadrants whose Shapley values were large enough
        """
        srs = []
        for i in range(len(shapley_values)):
            for j in range(len(shapley_values[0])):
                if shapley_values[i, j] > tol:
                    srs.append(regions[i, j])
        return srs

    def display_salient(self, im, srs_coll, count, filename ):
        """
        Determine which of the 4 quadrants are salient, i.e. have Shapley value larger than tol
        Parameters
        ----------
        im : the original image, in input format

        srs_coll : a collection of all regions deemed salient

        count : a normalizing mask which determines how many time each pixel was given a chance to be salient

        filename : name of the file to save the figure to
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 30))
        sample_image = im.numpy().transpose(1, 2, 0)
        count = count.transpose(1, 2, 0)
        image = sample_image * self.sd + self.mean
        ax1.imshow(image)
        ax2.imshow(image)
        mask = np.zeros(image.shape)

        # Count how many time each pixel was found to be in a salient region
        for srs in srs_coll:
            for sr in srs:
                start = sr[0]
                q_size = sr[1]

                xs = [start[1], start[1] + q_size[1], start[1] + q_size[1], start[1]]
                ys = [start[0], start[0], start[0] + q_size[0], start[0] + q_size[0]]
                ax2.fill(xs, ys, 'r', alpha=1 / len(srs_coll))
                mask[start[0]:start[0] + q_size[0], start[1]:start[1] + q_size[1], :] += np.ones(
                    (q_size[0], q_size[1], 3))

        # Normalize the mask by the number of tries in each region
        mask /= count
        # Normalize the mask to the range (0,1)
        mask /= np.max(mask)
        # Set to 0 elements smaller than 1/5
        negligible = (mask < 0.2)
        mask[negligible] = 0

        ax1.set_xlim([0, im.shape[2]])
        ax1.set_ylim([im.shape[1], 0])
        ax2.set_xlim([0, im.shape[2]])
        ax2.set_ylim([im.shape[1], 0])
        ax3.imshow(image * mask)
        if filename != None:
            plt.savefig(filename, dpi=300)

    def do_all(self, im, label, start, region_size, tol, debug=False):
        """
        Secondary main loop: do everything for one region of the image.
        ----------
        im : the input image

        start : the starting coordinates of the region

        region_size : self-explanatory

        tol : the specified tolerance for a sub-region to be considered salient

        debug : if True, all subsets, there associated scores and the Shapley values will be displayed

        Returns
        --------
        srs : a list of the coordinates of the quadrants whose Shapley values were large enough
        """
        images_final, regions = self.construct_subsets(im, start, region_size)
        score = self.subset_scores(images_final, label)
        sm = self.shapley_of_quadrants(score)
        if debug:
            self.display_cropped_images(images_final, score)
            f = plt.figure()
            sns.heatmap(sm)
            f.suptitle("Shap values of each quadrant")

        srs = self.get_salient_regions(sm, tol, regions)

        return srs

    def saliency_map(self, image, label, tolerance, only_one_run=False, debug=False, max_depth=30, filename=None):
        """
        Create and then show a saliency map built with the Hierarchical Shapley method.
        ----------
        im : the input image

        label : the label with respect to which we want to analyze - typically 1

        tolerance : the specified tolerance for a sub-region to be considered salient. A list is expected.

        only_one_run : when False, several runs are done by also considering 16 cropped versions of the input

        debug : if True, all subsets, there associated scores and the Shapley values will be displayed

        max_depth : the maximum number of divisions you want to allow before deciding the tolerance is too low.

        filename : name of the file to save the figure to
        """
        ls = []
        count = np.zeros(image.shape)
        xf = [image.shape[1], image.shape[2]]

        if only_one_run:
            starts = [(0, 0)]
            ends = [(xf[0], xf[1])]
        else:
            delta = [image.shape[1] // 20, image.shape[2] // 24]
            starts = [(0, 0), (0, delta[1]), (delta[0], 0), (delta[0], delta[1])]
            ends = [(xf[0], xf[1]), (xf[0], xf[1] - delta[1]), (xf[0] - delta[0], xf[1]),
                    (xf[0] - delta[0], xf[1] - delta[1])]

        for start in starts:
            for end in ends:
                size = (end[0] - start[0], end[1] - start[1])
                count[:, start[0]:end[0], start[1]:end[1]] += np.ones((3, size[0], size[1]))

        for tol in tolerance:
            try:
                for start in starts:
                    for end in ends:

                        size = (end[0] - start[0], end[1] - start[1])
                        srs = [(start, size)]
                        finished = []
                        k = 0
                        while len(srs) > 0:
                            if k > max_depth:
                                raise RuntimeError("Depth %d reached at tolerance %f" % (k, tol))
                            all_ = []
                            for sr in srs:
                                s = self.do_all(image, label, sr[0], sr[1], tol, debug)
                                if s == []:
                                    finished.append(((sr[0]), (sr[1])))
                                else:
                                    all_ += s
                            srs = all_
                            k += 1
                        ls.append(finished)
            except RuntimeError as w:
                print(w, "Run ignored, consider increasing tolerance.")

        self.display_salient(image, ls, count, filename)
