import torch.nn as nn
from torchvision import models



def build_model(num_classes: int, pretrained: bool = False):
    """
    Build the MobileNetV3-Small architecture used by TerraClass.

    During training:
        pretrained=True

    During inference:
        pretrained=False

    In inference, our locally trained model.pt supplies the weights.
    """

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.mobilenet_v3_small(weights= weights)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model



def freeze_backbone(model):
    """
    Freeze MobileNet feature extractor.

    This matches the training experiment.
    """

    for parameter in model.features.parameters():
        parameter.requires_grad = False


    return model