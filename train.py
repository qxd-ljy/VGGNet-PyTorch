import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vggnet import VGG
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


def get_data_loader(batch_size=64, num_workers=2):
    """
    获取 MNIST 数据集的 DataLoader

    参数:
    - batch_size (int): 每个 batch 的大小，默认 64
    - num_workers (int): 用于数据加载的线程数，默认 2

    返回:
    - DataLoader: 用于加载 MNIST 训练数据的 DataLoader 对象
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='D:/workspace/data', train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def initialize_model(device, num_classes=10):
    """
    初始化模型和优化器，并将其移动到指定设备

    参数:
    - device (torch.device): 使用的计算设备（CUDA 或 CPU）
    - num_classes (int): 分类任务中的类别数量，默认为 10（MNIST）

    返回:
    - model (torch.nn.Module): 初始化后的模型
    - optimizer (torch.optim.Optimizer): AdamW 优化器
    - criterion (torch.nn.Module): 交叉熵损失函数
    """
    model = VGG(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion


def train_epoch(model, train_loader, device, criterion, optimizer, scaler):
    """
    训练一个 epoch，并返回该 epoch 的平均损失和准确率

    参数:
    - model (torch.nn.Module): 训练的模型
    - train_loader (DataLoader): 用于加载训练数据的 DataLoader
    - device (torch.device): 当前模型所在的设备（CPU 或 GPU）
    - criterion (torch.nn.Module): 损失函数
    - optimizer (torch.optim.Optimizer): 优化器
    - scaler (GradScaler): 混合精度训练的梯度缩放器

    返回:
    - epoch_loss (float): 当前 epoch 的平均损失
    - epoch_accuracy (float): 当前 epoch 的准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc="Training", unit="batch", ncols=100) as pbar:
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 混合精度训练
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 更新进度条
            pbar.set_postfix(loss=running_loss / (total // len(data)), accuracy=100 * correct / total)

    return running_loss / len(train_loader), 100 * correct / total


def save_model(model, filepath='vggnet_mnist.pth'):
    """
    保存训练的模型到指定文件（覆盖之前的文件）

    参数:
    - model (torch.nn.Module): 训练完成后的模型
    - filepath (str): 保存模型的文件路径，默认 'vggnet_mnist.pth'
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def train_model(num_epochs=5, batch_size=64, lr=0.001):
    """
    训练模型的主函数，包含数据加载、模型初始化、训练和保存

    参数:
    - num_epochs (int): 训练的 epoch 数，默认为 5
    - batch_size (int): 每个 batch 的大小，默认为 64
    - lr (float): 学习率，默认为 0.001
    """
    # 获取数据加载器
    train_loader = get_data_loader(batch_size)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、优化器和损失函数
    model, optimizer, criterion = initialize_model(device)

    # 初始化混合精度训练的 GradScaler
    scaler = GradScaler()

    # 训练模型
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        epoch_loss, epoch_accuracy = train_epoch(model, train_loader, device, criterion, optimizer, scaler)

        # 打印每个 epoch 的损失和准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # 每个 epoch 结束时覆盖保存模型
        save_model(model)


if __name__ == "__main__":
    train_model()
