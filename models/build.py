from .resnet import ResNet101
from .resnet import ResNet50

def build_model(args) :
    model = ResNet50()
    return model