#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
import deepspeed
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    # standard DeepSpeed args: --deepspeed, etc.
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument(
        "--deepspeed_config_yaml",
        type=str,
        default="ds_config.yml",
        help="path to DeepSpeed YAML config"
    )
    return parser.parse_args()

class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

def main():
    args = get_args()

    # initialize with DeepSpeed
    with open(args.deepspeed_config_yaml, "r") as f:
        ds_cfg = yaml.safe_load(f)

    # prevent DeepSpeed from re-loading its own file
    args.deepspeed_config = None

    # set CUDA device based on local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local_rank)

    # MNIST DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.half())
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # build model
    model = SimpleMNIST()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_cfg
    )

    loss_fn = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(5):
        model_engine.train()
        total_loss = 0.0
        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()

            logits = model_engine(imgs)
            loss = loss_fn(logits, labels)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            if step % 200 == 0:
                print(f"[Epoch {epoch}] step {step}: loss {loss.item():.4f}")


        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch} complete. Avg loss: {avg:.4f}")

if __name__ == "__main__":
    main()
