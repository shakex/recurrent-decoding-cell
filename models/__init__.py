import copy
from models.FCN import FCN
from models.SegNet import SegNet
from models.UNet import UNet, VGGUNet, UNetFCN, UNetSegNet
from models.CRDN import UNetRNN, VGG16RNN, ResNet18RNN, ResNet50RNN, ResNet34RNN, ResNet101RNN, ResNet152RNN, ResNet50UNet, ResNet50FCN

"""
Implemented Models (15):
- FCN2s (FCN)
- SegNet (SegNet)
- U-Net (UNet)
- CRDN with VGG16 (VGG16RNN)
- CRDN with ResNet18/34/50/101/152 (ResNet18RNN/ResNet34RNN/ResNet50RNN/ResNet101RNN/ResNet152RNN)
- CRDN with U-Net-backbone (UNetRNN)
- U-Net with VGG16 (VGGUNet)
- U-Net with ResNet50 (ResNet50UNet)
- FCN with U-Net-backbone (UNetFCN)
- FCN with ResNet50 (ResNet50FCN)
- SegNet with U-Net-backbone (UNetSegNet)

"""

def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == "FCN":
        model = model(n_classes=n_classes, learned_billinear=False)
        # vgg16 = models.vgg16(pretrained=True)
        # model.init_vgg16_params(vgg16)

    elif name == "SegNet":
        model = model(n_classes=n_classes, in_channels=3, **param_dict)
        # vgg16 = models.vgg16(pretrained=True)
        # model.init_vgg16_params(vgg16)

    elif name == ["UNet","UNetFCN","UNetSegNet","VGGUNet"]:
        model = model(n_classes=n_classes, input_channel=3, **param_dict)

    elif name == "UNetRNN":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="LSTM", bias=True)

    elif name in ["VGG16RNN","ResNet18RNN", "ResNet34RNN", "ResNet50RNN", "ResNet101RNN", "ResNet152RNN"]:
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, decoder="GRU", bias=True)

    elif name in ["ResNet50UNet","ResNet50FCN"]:
        model = model(n_classes=n_classes, input_channel=3, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "FCN": FCN,
            "UNet": UNet,
            "UNetFCN": UNetFCN,
            "UNetSegNet": UNetSegNet,
            "VGGUNet": VGGUNet,
            "SegNet": SegNet,
            "UNetRNN": UNetRNN,
            "VGG16RNN": VGG16RNN,
            "ResNet18RNN": ResNet18RNN,
            "ResNet34RNN": ResNet34RNN,
            "ResNet50RNN": ResNet50RNN,
            "ResNet101RNN": ResNet101RNN,
            "ResNet152RNN": ResNet152RNN,
            "ResNet50UNet": ResNet50UNet,
            "ResNet50FCN": ResNet50FCN
        }[name]
    except:
        raise ("Model {} not available".format(name))

