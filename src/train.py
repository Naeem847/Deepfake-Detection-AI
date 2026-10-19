import argparse

import torch

import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .config import MODEL_PATH, CLASSES, FAKE_IDX
from .dataset import get_loaders
from .model import build_model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            probs += torch.softmax(out, 1)[:, FAKE_IDX].cpu().tolist()
            labels += (y == FAKE_IDX).int().cpu().tolist()
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    return correct / max(total, 1), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl = get_loaders(args.batch_size)
    assert train_dl.dataset.classes == CLASSES, f"Expected folders {CLASSES}"

    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    MODEL_PATH.parent.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        acc, auc = evaluate(model, val_dl, device)
        print(f"epoch {epoch}: loss={running/len(train_dl.dataset):.4f} val_acc={acc:.4f} val_auc={auc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  saved best model")


if __name__ == "__main__":
    main()