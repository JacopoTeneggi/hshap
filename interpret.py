import matplotlib.pyplot as plt
import seaborn as sns


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