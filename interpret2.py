import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils
import shap


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


def shapley_exp(e, inp, image):
    shapley_values, indexes = e.shap_values(inp, ranked_outputs=2, nsamples=200)
    shapley_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shapley_values]

    shap_red = [shapley_values[0][:, :, :, 0], shapley_values[1][:, :, :, 0]]
    shap_green = [shapley_values[0][:, :, :, 1], shapley_values[1][:, :, :, 1]]
    shap_blue = [shapley_values[0][:, :, :, 2], shapley_values[1][:, :, :, 2]]

    shap.image_plot(shapley_values, image, indexes.numpy(), show=False)
    plt.suptitle("All channels together");
    shap.image_plot(shap_red, image, indexes.numpy(), show=False)
    plt.suptitle("Red");
    shap.image_plot(shap_green, image, indexes.numpy(), show=False)
    plt.suptitle("Green");
    shap.image_plot(shap_blue, image, indexes.numpy(), show=False)
    plt.suptitle("Blue");

