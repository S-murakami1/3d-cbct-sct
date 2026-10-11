"""300-epoch MAE translation with dropout p=0.1 on encoder stages E3–E5."""
from typing import List, Tuple, Union

from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.dropout_arch import (
    build_network_deep_dropout,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae_scratch300 import (
    nnUNetTrainerMRCT_mae_scratch300,
)


class nnUNetTrainerMRCT_mae_scratch300_drop(nnUNetTrainerMRCT_mae_scratch300):
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return build_network_deep_dropout(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
