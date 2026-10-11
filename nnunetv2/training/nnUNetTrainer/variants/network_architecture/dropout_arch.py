"""Dropout p=0.1 on encoder stages E3–E5 only (not stem, not shallow encoder, not decoder)."""
from __future__ import annotations

from typing import List, Tuple, Union

from torch import nn

DROPOUT_P = 0.1
# ResidualUNet encoder: E0..E5. Keep dropout on E3, E4, E5 only.
ENCODER_DROPOUT_FROM = 3


def apply_dropout_kwargs(arch_init_kwargs: dict, p: float = DROPOUT_P) -> dict:
    kw = dict(arch_init_kwargs)
    conv = str(kw.get("conv_op", ""))
    kw["dropout_op"] = "torch.nn.Dropout2d" if "Conv2d" in conv else "torch.nn.Dropout3d"
    kw["dropout_op_kwargs"] = {"p": float(p), "inplace": False}
    return kw


def _zero_dropout(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0


def iter_dropout(network: nn.Module):
    for name, m in network.named_modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            yield name, m


def restrict_dropout_to_deep_encoder(network: nn.Module, keep_from: int = ENCODER_DROPOUT_FROM) -> nn.Module:
    """Disable dropout everywhere except encoder.stages[keep_from:].

    ResidualUNet decoder stores ``self.encoder`` as a child. Zeroing the whole
    decoder would walk that child and also kill E3–E5. Skip the encoder alias.
    """
    enc = getattr(network, "encoder", None)
    if enc is None:
        return network
    if getattr(enc, "stem", None) is not None:
        _zero_dropout(enc.stem)
    stages = getattr(enc, "stages", None)
    if stages is not None:
        for i, stage in enumerate(stages):
            if i < keep_from:
                _zero_dropout(stage)
    dec = getattr(network, "decoder", None)
    if dec is not None:
        for name, child in dec.named_children():
            if name == "encoder" or child is enc:
                continue
            _zero_dropout(child)
    return network


def build_network_deep_dropout(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
) -> nn.Module:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae import (
        nnUNetTrainerMRCT_mae,
    )

    net = nnUNetTrainerMRCT_mae.build_network_architecture(
        architecture_class_name,
        apply_dropout_kwargs(arch_init_kwargs),
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    net = restrict_dropout_to_deep_encoder(net)
    enc = getattr(net, "encoder", None)
    per_stage = []
    n_pos = 0
    if enc is not None and getattr(enc, "stages", None) is not None:
        for i, st in enumerate(enc.stages):
            n = sum(1 for _, m in iter_dropout(st) if float(m.p) > 0)
            per_stage.append(f"E{i}:{n}")
            n_pos += n
    print(
        f"deep dropout: p={DROPOUT_P} from E{ENCODER_DROPOUT_FROM} "
        f"({', '.join(per_stage)}; {n_pos} modules p>0). "
        "plans.json dropout_op=None is expected."
    )
    return net
