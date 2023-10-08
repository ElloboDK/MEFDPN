import logging
from typing import Any, Dict, Optional
import pandas as pd
import torch
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import StandardScaler, tabular_train_test_split, tabular_ood_dataset

logger = logging.getLogger(__name__)


@register("mydata")
class BankDataModule(DataModule):
    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.did_setup = False

        self.input_scaler = StandardScaler()

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([10])
    
    @property
    def num_classes(self) -> int:
        return 2

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            df = pd.read_csv("D:/Postgraduate/106/Uncertainty/code/natural-posterior-network-main/natpn/datasets/bank.csv")
            
            X_base = torch.from_numpy(df.to_numpy()[:, 1:-1]).float()
            y_base = torch.from_numpy(df.to_numpy()[:, -1]).long()

            # Split by classes
            ood_mask = (y_base == 1)
            X, X_ood = X_base[~ood_mask], X_base[ood_mask]
            y, _ = y_base[~ood_mask], y_base[ood_mask]

            # Split data
            (X_train, X_test), (y_train, y_test) = tabular_train_test_split(
                X, y, train_size=0.8, generator=self.generator
            )
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
                        
            # Fit transforms
            self.input_scaler.fit(X_train)

            # Create datasets
            self.train_dataset = TensorDataset(self.input_scaler.transform(X_train), y_train)
            self.val_dataset = TensorDataset(self.input_scaler.transform(X_val), y_val)
            self.test_dataset = TensorDataset(self.input_scaler.transform(X_test), y_test)
            self.ood_datasets = {
                "ood": tabular_ood_dataset(
                    self.input_scaler.transform(X_test), self.input_scaler.transform(X_ood)
                ),
            }

            # Mark done
            self.did_setup = True


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=512)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=512)
    
    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=4096)
            for name, dataset in self.ood_datasets.items()
        }