import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import Net 
from visdom_helper import VisdomHelper
from torchsummary import summary


def evaluate(net, loader, device):
    """Evaluate accuracy on a dataset."""
    net.eval()
    num_correct, num_total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)

    return num_correct / num_total


def train(args):
    # -------------------------
    # Prepare dataset
    # -------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------
    # Device, model, loss, optimizer
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # -------------------------
    # Initialize Visdom
    # -------------------------
    viz = VisdomHelper(server="http://localhost", port=8097)
    viz.show_text("Training started!", title="Status")

    # model summary via torchsummary
    summary(net, (1, 28, 28))
    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(args.max_epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            # Log every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                step = epoch * len(train_loader) + batch_idx
                viz.plot_scalar(avg_loss, step, title="Training Loss", xlabel="Step", ylabel="Loss")
                print(f"Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.5f}")
                running_loss = 0.0

        # End of epoch stats
        train_acc = correct_train / total_train
        test_acc = evaluate(net, test_loader, device)
        print(f"\nEpoch [{epoch+1}/{args.max_epochs}] Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}\n")
        viz.show_text(f"Epoch {epoch+1}/{args.max_epochs} complete. Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}", title="Status")
        viz.plot_scalar(train_acc, epoch, title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
        viz.plot_scalar(test_acc, epoch, title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy")

    # -------------------------
    # Save model
    # -------------------------
    torch.save(net.state_dict(), "mnist-final.pth")
    viz.show_text("Training complete! Model saved as mnist-final.pth", title="Status")
    print("Training finished and model saved.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MNIST training with Visdom logging")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
