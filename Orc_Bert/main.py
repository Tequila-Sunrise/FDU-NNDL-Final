import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T

from matplotlib import pyplot as plt

import os
import argparse


parser = argparse.ArgumentParser(description="OrcBert Training")
parser.add_argument(
    '--dir', type=str, default='data/oracle_1shot', help='data directory'
)
parser.add_argument(
    '--name', type=str, default='oracle_1shot', help='name of the model'
)
parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--max-epoch', type=int, default=200, help="Max training epochs")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # Best test accuracy

# Data
print('==> Preparing data...')
transform_train = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(45, fill=255, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize((0.84,), (0.32,)),
    ]
)

transform_test = T.Compose(
    [T.Resize(224), T.ToTensor(), T.Normalize((0.84,), (0.32,)),]
)

train_set = torchvision.datasets.ImageFolder(
    root=os.path.join(args.dir, 'train'), transform=transform_train
)
train_batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=train_batch_size, shuffle=True, num_workers=8
)

test_set = torchvision.datasets.ImageFolder(
    root=os.path.join(args.dir, 'test'), transform=transform_test
)
test_batch_size = 256
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=test_batch_size, shuffle=False, num_workers=8
)

# Model
print(f"==> Building model...")
net = torchvision.models.resnet18(num_classes=200).to(device)
if device == 'cuda':
    net = nn.DataParallel(net)
    cudnn.benchmark = True

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = train_loss / ((batch_idx + 1) * train_batch_size)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = test_loss / ((batch_idx + 1) * test_batch_size)
    epoch_acc = 100.0 * correct / total

    # Save checkpoint
    if epoch_acc > best_acc:
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(net.state_dict(), f'./checkpoints/checkpoint_{args.name}.pth')
        best_acc = epoch_acc

    return epoch_loss, epoch_acc


# Training
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(args.max_epoch):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()

    # Print log info
    print("============================================================")
    print(f"Epoch: {epoch + 1}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%"
    )
    print("============================================================")

    # Logging
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)


# Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='train')
plt.plot(test_accuracies, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

if not os.path.isdir('log'):
    os.mkdir('log')
plt.savefig(f'./log/{args.name}.png')
