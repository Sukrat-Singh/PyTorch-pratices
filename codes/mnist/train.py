import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import Net  # Make sure net.py is in the same folder


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
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer
    net = Net().to(device)
    loss_op = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.max_epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = loss_op(outputs, labels)

            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Track batch statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            # Print every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.5f}")

        # Epoch statistics
        train_acc = correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        test_acc = evaluate(net, test_loader, device)

        print("\n")
        print(f"Epoch [{epoch+1}/{args.max_epochs}] "
              f"Avg Loss: {avg_loss:.5f} "
              f"Train Acc: {train_acc:.3f} "
              f"Test Acc: {test_acc:.3f}")
        print("\n")
        print("----------------------------------------------------------------------------------")
        print("\n")

    # Save model
    torch.save(net.state_dict(), "mnist-final.pth")
    print("DONE!  Model saved as mnist-final.pth\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MNIST training script with more info")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of training epochs")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
