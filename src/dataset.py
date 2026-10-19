from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DATA_DIR, IMG_SIZE, MEAN, STD


def get_transforms(train: bool):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def get_loaders(batch_size: int = 32, num_workers: int = 2):
    train_ds = datasets.ImageFolder(DATA_DIR / "train", get_transforms(True))
    val_ds = datasets.ImageFolder(DATA_DIR / "val", get_transforms(False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    return train_dl, val_dl