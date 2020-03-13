import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from gradcam.utils import visualize_cam
import os
os.system("wget https://raw.githubusercontent.com/yiskw713/SmoothGradCAMplusplus/master/utils/visualize.py -P local_modules -nc")
import sys
sys.path.append('local_modules')
from local_modules.visualize import visualize


def display_gradients(gradients, figure_size=(6, 5)):
    fig = plt.figure(figsize=(3*figure_size[0], figure_size[1]))
    ax1 = fig.add_subplot(131, title="Gradient in the red channel")
    ax2 = fig.add_subplot(132, title="Gradient in the green channel")
    ax3 = fig.add_subplot(133, title="Gradient in the blue channel")
    _ = sns.heatmap(gradients[0], cmap="Reds", ax=ax1)
    _ = sns.heatmap(gradients[1], cmap="Greens", ax=ax2)
    _ = sns.heatmap(gradients[2], cmap="Blues", ax=ax3)
    return fig


def shap_exp(e, inp, img):
    shapley_values, indexes = e.shap_values(inp, ranked_outputs=2, nsamples=200)
    shapley_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shapley_values]

    shap_red = [shapley_values[0][:, :, :, 0], shapley_values[1][:, :, :, 0]]
    shap_green = [shapley_values[0][:, :, :, 1], shapley_values[1][:, :, :, 1]]
    shap_blue = [shapley_values[0][:, :, :, 2], shapley_values[1][:, :, :, 2]]

    image = img[np.newaxis, :]

    shap.image_plot(shapley_values, image, indexes.numpy(), show=False)
    plt.suptitle("All channels together");
    shap.image_plot(shap_red, image, indexes.numpy(), show=False)
    plt.suptitle("Red");
    shap.image_plot(shap_green, image, indexes.numpy(), show=False)
    plt.suptitle("Green");
    shap.image_plot(shap_blue, image, indexes.numpy(), show=False)
    plt.suptitle("Blue");


def gradcam_exp(gradcam, gradcam_pp, inp, image, layer_name, f_size):
    mask, _ = gradcam(inp)
    heatmap, result = visualize_cam(mask, inp)
    mask_pp, _ = gradcam_pp(inp)
    heatmap_pp, result_pp = visualize_cam(mask_pp, inp)
    display_gradients(heatmap.detach().numpy(), f_size).suptitle("Grad-CAM for an image with label 0", size="xx-large")
    display_gradients(heatmap_pp.detach().numpy(), f_size).suptitle("Grad-CAM++ for an image with label 0", size="xx-large")

    #heatmap_show = np.swapaxes(np.swapaxes(heatmap, 0, 1), 1, 2)
    #heatmap_pp_show = np.swapaxes(np.swapaxes(heatmap_pp, 0, 1), 1, 2)
    result_show = np.swapaxes(np.swapaxes(result.detach(), 0, 1), 1, 2)
    result_pp_show = np.swapaxes(np.swapaxes(result_pp.detach(), 0, 1), 1, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*f_size[0], f_size[1]))

    ax1.imshow(result_show)
    ax1.set_title("With GradCAM")
    ax2.imshow(result_pp_show)
    ax2.set_title("With GradCAM++")
    fig.suptitle("With respect to %s" % layer_name, size="xx-large")


def smooth_exp(inp, image, wrapped): 
  fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,20))
  titles = ["GradCAM","GradCAM++","SmoothGradCAM++"]
  for i in range(len(wrapped)):
    cam, idx = wrapped[i](inp)
    heatmap = visualize(image.unsqueeze(0), cam)
    hm = (heatmap.squeeze().detach().numpy().transpose(1, 2, 0))
    ax[0][i].imshow(cam.squeeze().numpy(), alpha=0.5, cmap='jet')
    ax[1][i].imshow(hm)
    ax[1][i].set_title(titles[i])
