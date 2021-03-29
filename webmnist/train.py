from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from typing import NamedTuple
from tqdm import tqdm
from webmnist.model import LeNet5

import torch
import torch.nn as nn
import torchvision.transforms as T


MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST.resources = [
   ("/".join([MIRROR, url.split("/")[-1]]), md5)
   for url, md5 in MNIST.resources
]


class Datasets(NamedTuple):
    train = MNIST(
        "data/mnist",
        download=True,
        train=True,
        transform=T.Compose([T.RandomRotation(20), T.ToTensor()],
    ))
    test = MNIST(
        "data/mnist",
        download=True,
        train=False,
        transform=T.ToTensor(),
    )


class Loaders(NamedTuple):
    train: DataLoader
    test: DataLoader


def train(path: str, epochs: int = 5) -> None:
    datasets = Datasets()
    loaders = Loaders(
        train=DataLoader(
            datasets.train,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
        ),
        test=DataLoader(
            datasets.test,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
        ),
    )

    model = LeNet5(n_classes=10).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optim = AdamW(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        total_loss, total_acc = 0, 0
        
        pbar = tqdm(loaders.train, desc="Train")
        for img, label in pbar:
            img, label = img.cuda(), label.cuda()
            optim.zero_grad()
            
            pred = model(img)
            loss = criterion(pred, label)
            acc = (torch.argmax(pred, dim=1) == label).sum()

            loss.backward()
            optim.step()

            total_loss += loss.item() / len(loaders.train)
            total_acc += acc.item() / len(datasets.train)
            
            pbar.set_postfix(
                loss=f"{total_loss:.2e}",
                acc=f"{total_acc * 100:.2f}%",
            )

        model.eval()
        total_loss, total_acc = 0, 0
        
        pbar = tqdm(loaders.test, desc="Test")
        for img, label in pbar:
            img, label = img.cuda(), label.cuda()
            optim.zero_grad()
            
            pred = model(img)
            loss = criterion(pred, label)
            acc = (torch.argmax(pred, dim=1) == label).sum()

            loss.backward()
            optim.step()

            total_loss += loss.item() / len(loaders.test)
            total_acc += acc.item() / len(datasets.test)
            
            pbar.set_postfix(
                loss=f"{total_loss:.2e}",
                acc=f"{total_acc * 100:.2f}%",
            )

    torch.save(model.state_dict(), f"{path}.pth")