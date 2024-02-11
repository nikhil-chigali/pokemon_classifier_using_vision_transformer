import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from configs import data_config


def visualize_batch(images, targets, preds=None):
    images = images.type(torch.DoubleTensor)
    targets = targets.type(torch.ByteTensor)
    if preds is not None:
        preds = preds.type(torch.ByteTensor)

    images = images * data_config.std.view(1, -1, 1, 1) + data_config.mean.view(
        1, -1, 1, 1
    )
    images = images.numpy().transpose((0, 2, 3, 1))

    n = images.shape[0]

    _, axes = plt.subplots(nrows=2, ncols=n // 2)
    axes = list(axes[0]) + list(axes[1])
    for i in range(n):
        img = to_pil_image(images[i], mode="RGB")
        axes[i].imshow(img)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        if preds is not None:
            axes[i].set_xlabel(
                f"y: {data_config.idx_to_class[targets[i].item()]}\ny_hat: {data_config.idx_to_class[preds[i].item()]}"
            )
        else:
            axes[i].set_xlabel(f"y: {data_config.idx_to_class[targets[i].item()]}")

    plt.show()
