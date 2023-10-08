from ._base import DataModule, OutputType
from ._registry import DATASET_REGISTRY
from .bike_sharing import BikeSharingNormalDataModule, BikeSharingPoissonDataModule
from .cifar import Cifar10DataModule, Cifar100DataModule
from .mnist import FashionMnistDataModule, MnistDataModule
from .nyu_depth_v2 import NyuDepthV2DataModule
from .sensorless_drive import SensorlessDriveDataModule
from .uci import ConcreteDataModule
from .mydata import MyDataModule
from .CelebA import CelebADataModule
from .AID362 import AID362DataModule
from .ad import adDataModule
from .chess import chessDataModule
from .bank import BankDataModule
from .census import CensusDataModule
from .Probe import ProbeDataModule
from .U2R import U2RDataModule
from .NIMS import NIMSDataModule
from .DMax import DMaxDataModule

__all__ = [
    "CelebADataModule",
    "chessDataModule",
    "BankDataModule",
    "CensusDataModule",
    "NIMSDataModule",
]
