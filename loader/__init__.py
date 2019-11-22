from loader.brainwebLoader import brainwebLoader
from loader.pascal_voc_loader import pascalVOCLoader
from loader.MSRC24Loader import msrc24Loader
from loader.petLoader import petLoader
from loader.celebALoader import celebALoader
from loader.neobrains12Loader import neobrains12Loader
from loader.mrbrainsLoader import mrbrainsLoader
from loader.hvsmrLoader import hvsmrLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "VOC2012": pascalVOCLoader,
        "BrainWeb": brainwebLoader,
        "MSRC24": msrc24Loader,
        "Pet": petLoader,
        "CelebA": celebALoader,
        "NeoBrainS12": neobrains12Loader,
        "MRBrainS": mrbrainsLoader,
        "HVSMR": hvsmrLoader
    }[name]