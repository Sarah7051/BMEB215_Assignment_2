import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


DATA_DIR = "./dataset/train"
LOG_DIR = "./logs/epilepsy_experiment"
WEIGHTS_DIR = "./weights"

NUM_FRAMES = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_CLASSES = 2
VAL_SPLIT = 0.2

torch.manual_seed(42)
np.random.seed(42)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}", flush=True)


class EpilepsyVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=10):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.classes = ["n", "y"]
        self.data = []

        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_path):
                continue
            for vid_name in os.listdir(cls_path):
                if vid_name.endswith((".mp4", ".avi", ".mov")):
                    self.data.append((os.path.join(cls_path, vid_name), cls_idx))

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.zeros(self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        return torch.stack(frames[: self.num_frames])

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        return self._load_video(video_path), label


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class VideoResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(batch_size, time_steps, -1)
        x = torch.mean(x, dim=1)
        return self.fc(x)


data_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

full_dataset = EpilepsyVideoDataset(DATA_DIR, transform=data_transforms, num_frames=NUM_FRAMES)
total_size = len(full_dataset)
val_size = int(total_size * VAL_SPLIT)
train_size = total_size - val_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"dataset: total={total_size}, train={len(train_dataset)}, val={len(val_dataset)}", flush=True)


def train_model(model, model_name, epochs=20):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(os.path.join(LOG_DIR, model_name))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predicted = torch.max(outputs.data, 1)[1]
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = torch.max(outputs.data, 1)[1]
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val

        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", epoch_val_acc, epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}",
            flush=True,
        )

        if (epoch + 1) == 10 or (epoch + 1) == epochs:
            save_path = os.path.join(WEIGHTS_DIR, f"{model_name}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"saved: {save_path}", flush=True)

    writer.close()


if __name__ == "__main__":
    train_model(VideoResNet18(num_classes=NUM_CLASSES), "resnet18", epochs=EPOCHS)
