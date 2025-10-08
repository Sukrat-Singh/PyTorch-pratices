import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import Net  # changed from .net to net


def evaluate(net, loader, device):
    net.eval()
    num_correct, num_total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, preds = torch.max(outputs, 1)

            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)

    return num_correct / num_total


def train(args):
    # create transforming pipeline: convert to tensor + normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # prepare the MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # turn on CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net().to(device)
    loss_op = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.max_epochs):
        net.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward propagation
            outputs = net(images)
            loss = loss_op(outputs, labels)

            # back-prop
            optim.zero_grad()
            loss.backward()
            optim.step()

        acc = evaluate(net, test_loader, device)
        print(f"Epoch [{epoch+1}/{args.max_epochs}] loss: {loss.item():.5f} test acc: {acc:.3f}")

    torch.save(net.state_dict(), "mnist-final.pth")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of training epochs")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
