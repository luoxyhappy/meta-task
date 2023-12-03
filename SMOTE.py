#用于跑SMOTE算法
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from PIL import Image

for classid in range(0,10):
    # 定义数据集路径和保存路径
    data_dir = "/home/demo2/FL_data/cifar-10/cifar-10-batches-py"
    save_dir = "/home/demo2/Slaugfl_results/oversample/{}/0".format(classid)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 定义数据预处理的转换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载原始CIFAR-10数据集
    cifar_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    # 获取原始的0类图像和标签
    class_0_images = []
    y = []
    for image, label in cifar_dataset:
        class_0_images.append(image)
        if label == classid:
            y.append(0)
        else:
            y.append(1)


    class_0_images=torch.stack(class_0_images)
    class_0_images = np.array(class_0_images)
    y = np.array(y)


    # 使用SMOTE算法生成合成样本
    smote = BorderlineSMOTE(kind="borderline-2")
    print(class_0_images.shape)
    oversampled_images, _ = smote.fit_resample(class_0_images.reshape(-1, 32 * 32 * 3), y )
    oversampled_images = oversampled_images.reshape(-1, 3, 32 ,32)
    print(oversampled_images.shape)
    oversampled_images = np.transpose(oversampled_images, (0 ,2, 3, 1))
    # 将合成样本转换回PIL图像并保存为PNG文件
    m = 0
    for i, (image,label)in enumerate(zip(oversampled_images,_)):
        if label == 0:
            image = Image.fromarray((image * 255).astype(np.uint8))
            save_path = os.path.join(save_dir, f"image_{i}.png")
            image.save(save_path)
            m = m + 1

    print(m)
    print(f"Saved {len(oversampled_images)} images to {save_dir} directory.")