"""Model architectures and preprocessing shared by training and inference."""
from torch import nn
from torchvision import models, transforms as T

IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Every architecture below can be trained and compared in MLflow.
# "simple_cnn" is trained from scratch (baseline), the others use ImageNet transfer learning.
SUPPORTED_MODELS = (
    "simple_cnn",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "convnext_tiny",
)
# Models used when nobody picks any (fast enough for a full benchmark on one laptop).
DEFAULT_BENCHMARK_MODELS = ("simple_cnn", "mobilenet_v3_small", "resnet18", "efficientnet_b0")

def evaluation_transform(size: int = IMG, mean=MEAN, std=STD):
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(mean, std)])


def build_transforms(size: int = IMG):
    """(train, eval) transforms for a given square input size. The app reads the size from the checkpoint."""
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(0.4, 0.4, 0.3, 0.05),
        T.RandomApply([T.GaussianBlur(5)], p=0.3),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    return train, evaluation_transform(size)


class SimpleCNN(nn.Module):
    """Small CNN trained from scratch: the baseline every pretrained model must beat."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(num_classes: int, model_name: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        if model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
        else:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")
