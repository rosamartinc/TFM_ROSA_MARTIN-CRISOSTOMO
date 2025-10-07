from .sincnet import SincNetECG
from .cnn_baseline import CNNBaseline1D
from .resnet1d import ResNet1D
from .inception1d import Inception1D

_REGISTRY = {
    'sincnet': SincNetECG,
    'cnn_baseline': CNNBaseline1D,
    'resnet1d': ResNet1D,
    'inception1d': Inception1D,
}

def build_model(name, n_classes, **kwargs):
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Modelo '{name}' no registrado. Disponibles: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](n_classes=n_classes, **kwargs)
