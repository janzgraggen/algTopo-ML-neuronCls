"""

THINK WHETHRE THIS SHOULD BE DONE VIA CLI. 

...

"""


# src/train/scaled_trainer.py

import numpy as np
from copy import deepcopy
from typing import Sequence, Optional

import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from morphoclass.transforms.scalers import FeatureStandardScaler
from morphoclass.data import MorphologyDataLoader, MorphologyDataset
from morphoclass.training.trainers import Trainer 


def dataset_prep_fn(): 
    raise NotImplementedError


class ScaledTrainer(Trainer):
    """
    Extends BaseTrainer to offer single‑split and cross‑validation
    training with per‑split FeatureStandardScaler.
    """

    def _scale_subset(
        self,
        idx: Sequence[int],
        feature_indices: Sequence[int],
    ):
        """Helper: returns a new dataset view with scaler.fit on idx, then assigned as .transform."""
        ds_sub = self.dataset.index_select(idx)
        scaler = FeatureStandardScaler(feature_indices=feature_indices)
        scaler.fit(ds_sub)
        ds_sub.transform = scaler
        return ds_sub

    def train_single_split(
        self,
        split_ratio: float = 0.8,
        n_epochs: int = 100,
        batch_size: int = 16,
        load_best: bool = True,
        feature_indices: Optional[Sequence[int]] = None,
    ):
        N = len(self.dataset)
        split = int(N * split_ratio)
        train_idx = list(range(split))
        val_idx   = list(range(split, N))

        if feature_indices is None:
            feature_indices = list(range(self.dataset[0].x.shape[1]))

        # build scaled subsets
        train_ds = self._scale_subset(train_idx, feature_indices)
        val_ds   = self._scale_subset(val_idx,   feature_indices)

        # combine for BaseTrainer API
        from torch.utils.data import ConcatDataset
        combo = MorphologyDataset(ConcatDataset([train_ds, val_ds]))
        train_idx_c = list(range(len(train_ds)))
        val_idx_c   = list(range(len(train_ds), len(train_ds) + len(val_ds)))

        # swap dataset
        orig = self.dataset
        self.dataset = combo
        history = super().train(
            n_epochs=n_epochs,
            batch_size=batch_size,
            train_idx=train_idx_c,
            val_idx=val_idx_c,
            load_best=load_best,
        )
        self.dataset = orig
        return history

    def train_crossval(
        self,
        n_splits: int = 5,
        n_epochs: int = 100,
        batch_size: int = 16,
        load_best: bool = False,
        feature_indices: Optional[Sequence[int]] = None,
        random_state: int = 42,
        custom_splits =  None,
    ):
        """
        Cross‑validate using BaseTrainer.train for each fold, with per‑fold scaling.

        - custom_splits: if you already have (train_idx, val_idx) arrays, pass them here.
        """
        labels = np.array(self.dataset.labels)

        if feature_indices is None:
            feature_indices = list(range(self.dataset[0].x.shape[1]))

        # Prepare folds
        if custom_splits is None:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(skf.split(np.zeros(len(labels)), labels))
        else:
            splits = custom_splits  # already list of (train_idx, val_idx)

        histories = []
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            print(f"▶ Fold {fold}/{len(splits)}")

            # make fresh copies of net & optimizer
            net_copy = deepcopy(self.net).to(self.device)
            opt_copy = torch.optim.Adam(net_copy.parameters(), lr=self.optimizer.defaults['lr'])
            
            # new Trainer for this fold
            trainer = Trainer(
                net=net_copy,
                dataset=self.dataset,       # we'll swap in scaled combo below
                optimizer=opt_copy,
                loader_class=MorphologyDataLoader,
            )

            # scale per‑fold
            train_ds = self._scale_subset(train_idx, feature_indices)
            val_ds   = self._scale_subset(val_idx,   feature_indices)

            from torch.utils.data import ConcatDataset
            combo = ConcatDataset([train_ds, val_ds])
            train_idx_c = list(range(len(train_ds)))
            val_idx_c   = list(range(len(train_ds), len(train_ds) + len(val_ds)))

            # swap dataset
            orig = trainer.dataset
            trainer.dataset = combo

            hist = trainer.train(
                n_epochs=n_epochs,
                batch_size=batch_size,
                train_idx=train_idx_c,
                val_idx=val_idx_c,
                load_best=load_best,
            )
            histories.append(hist)

            # restore
            trainer.dataset = orig

        return histories
