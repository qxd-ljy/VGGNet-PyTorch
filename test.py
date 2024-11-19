import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vggnet import VGG


def get_test_loader(batch_size=64, data_dir='D:/workspace/data'):
    """ 获取 MNIST 测试数据加载器 """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def load_model(model_path='vggnet_mnist.pth', num_classes=10):
    """ 加载预训练模型 """
    model = VGG(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


def evaluate_model(model, test_loader, device):
    """ 评估模型并返回准确率和前六张图片的预测与标签 """
    model.eval()
    correct = 0
    total = 0
    images, labels, preds = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 记录前六张图片及其标签和预测
            if len(images) < 6:
                batch_size = data.size(0)
                for i in range(min(6 - len(images), batch_size)):
                    images.append(data[i].cpu())
                    labels.append(target[i].cpu())
                    preds.append(predicted[i].cpu())

    accuracy = 100 * correct / total
    return accuracy, images, labels, preds


def display_images(images, labels, preds):
    """ 可视化前六张图片及其真实标签和预测标签 """
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()

    for i in range(6):
        axes[i].imshow(images[i][0].squeeze(), cmap='gray')  # MNIST 是单通道灰度图像
        axes[i].set_title(f"True: {labels[i].item()}, Pred: {preds[i].item()}")
        axes[i].axis('off')  # 不显示坐标轴

    plt.show()


def test_model(model_path='vggnet_mnist.pth', data_dir='D:/workspace/data', batch_size=64):
    """ 主函数：测试模型并显示结果 """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据加载器
    test_loader = get_test_loader(batch_size, data_dir)

    # 加载模型
    model = load_model(model_path)
    model.to(device)

    # 评估模型
    accuracy, images, labels, preds = evaluate_model(model, test_loader, device)

    # 输出测试准确率
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 可视化结果
    display_images(images, labels, preds)


if __name__ == "__main__":
    test_model()
