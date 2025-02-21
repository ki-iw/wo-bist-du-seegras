import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as F


def convert_msec(msec):
    """Convert milliseconds to hours, minutes, seconds, and milliseconds."""
    milliseconds = int((msec % 1000) / 100)
    seconds = int(msec / 1000) % 60
    minutes = int(msec / (1000 * 60)) % 60
    hours = int(msec / (1000 * 60 * 60)) % 24
    return hours, minutes, seconds, milliseconds


def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(
        0,
        1,
    )


def single_image_preds(preds, whole_image, image_name):
    whole_image = whole_image.cpu().permute(1, 2, 0).numpy()
    # TODO get this from config or somewhere else
    class_list = ["BG", "Ferny", "Rounded", "Strappy"]

    fig = plt.figure(dpi=200)
    fig.set_size_inches(20, 10)
    ax = fig.add_subplot()

    plt.axis("off")
    ax.imshow(np.squeeze(whole_image), alpha=1.0)

    scale = np.array([np.shape(whole_image)[1], np.shape(whole_image)[0]])

    sigmoid_vals = preds.detach().cpu().numpy()

    if np.shape(sigmoid_vals)[0] == 45:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 9))  # rows, columns
    elif np.shape(sigmoid_vals)[0] == 35:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 7))  # rows, columns
    elif np.shape(sigmoid_vals)[0] == 40:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 8))  # rows, columns
    elif np.shape(sigmoid_vals)[0] == 54:
        sigmoid_vals = np.reshape(sigmoid_vals, (6, 9))  # rows, columns
    elif np.shape(sigmoid_vals)[0] == 77:
        sigmoid_vals = np.reshape(sigmoid_vals, (7, 11))  # rows, columns
    elif np.shape(sigmoid_vals)[0] == 6:
        sigmoid_vals = np.reshape(sigmoid_vals, (2, 3))  # rows, columns
    else:
        print("Not sure how many rows and columns!")

    CMAP = [[255, 20, 147], [255, 0, 0], [0, 0, 255], [255, 165, 0]]
    CMAP = np.asarray(CMAP)
    colour_predictions = CMAP[sigmoid_vals]

    offs = np.array([scale[0] / sigmoid_vals.shape[1], scale[1] / sigmoid_vals.shape[0]])
    for pos, val in np.ndenumerate(sigmoid_vals):
        ax.annotate(class_list[val], xy=np.array(pos)[::-1] * offs + offs / 2, ha="center", va="center", fontsize=15)

    heatmap = ax.imshow(np.flipud(colour_predictions), alpha=0.1, aspect="auto", extent=(0, scale[0], 0, scale[1]))  # noqa: F841

    ax.invert_yaxis()

    plt.savefig(image_name, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def img_to_grid(img, row, col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]), col)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]), row)]
    grid = [img[h[0] : h[1] + 1, w[0] : w[1] + 1, :] for h in hh for w in ww]
    return grid, len(hh), len(ww)


def cropper(image, width, height):
    if isinstance(image, np.ndarray):
        image = torch.tensor(image).permute(2, 0, 1)
    return F.center_crop(image, output_size=(height, width))
