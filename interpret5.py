import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
!pip install pytorch-gradcam
from gradcam.visualize import visualize_cam


def display_gradients( image, gradients, figure_size=(6, 5)):
    fig = plt.figure(figsize=(4*figure_size[0]*1.2, figure_size[1]))
    ax0 = fig.add_subplot(141, title="Input image")
    ax1 = fig.add_subplot(142, title="Gradient in the red channel")
    ax2 = fig.add_subplot(143, title="Gradient in the green channel")
    ax3 = fig.add_subplot(144, title="Gradient in the blue channel")
    ax0.imshow(image)
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
    display_gradients(image, heatmap.detach().numpy()).suptitle("Grad-CAM for an image with label 0", size="xx-large")
    display_gradients(image, heatmap_pp.detach().numpy()).suptitle("Grad-CAM++ for an image with label 0", size="xx-large")

    heatmap_show = np.swapaxes(np.swapaxes(heatmap, 0, 1), 1, 2)
    heatmap_pp_show = np.swapaxes(np.swapaxes(heatmap_pp, 0, 1), 1, 2)
    result_show = np.swapaxes(np.swapaxes(result.detach(), 0, 1), 1, 2)
    result_pp_show = np.swapaxes(np.swapaxes(result_pp.detach(), 0, 1), 1, 2)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=f_size)
    ax1.imshow(heatmap_show)
    ax1.set_title("With GradCAM")
    ax2.imshow(result_show)
    ax2.set_title("With GradCAM")
    ax3.imshow(heatmap_pp_show)
    ax3.set_title("With GradCAM++")
    ax4.imshow(result_pp_show)
    ax4.set_title("With GradCAM++")
    fig.suptitle("With respect to %s" % layer_name, size="xx-large")


