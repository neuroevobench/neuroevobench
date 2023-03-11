def normalize(data_tensor):
    """re-scale image values to [-1, 1]"""
    return (data_tensor / 255.0) * 2.0 - 1.0


def get_svhn_loaders(test: bool = False):
    """Get PyTorch Data Loaders for SVHN."""
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `VisionFitness` module."
        )

    transform = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: normalize(x)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ]

    bs = 26032 if test else 73257
    loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root="~/data", train=not test, download=True, transform=transform
        ),
        batch_size=bs,
        shuffle=False,
    )
    return loader
