def get_cifar_loaders(test: bool = False):
    """Get PyTorch Data Loaders for CIFAR-10."""
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `VisionFitness` module."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    bs = 10000 if test else 50000
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="~/data", train=not test, download=True, transform=transform
        ),
        batch_size=bs,
        shuffle=False,
    )
    return loader
