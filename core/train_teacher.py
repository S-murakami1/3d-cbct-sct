"""Train the 7-class AFP teacher (teacher_dataset_id) with the stock nnU-Net.

Run from the repository root in the TotalSegmentator environment:

    uv run --project totalseg python core/train_teacher.py

This repository's nnunetv2 and its batchgenerators fork interpolate the
segmentation slot like an image (CT HU for translation). Integer labels become
-3..9 under rotation and scaling, and the Dice one-hot scatter fails on the GPU.
The stock nnunetv2 in totalseg/.venv interpolates labels per class, so the
teacher trains there. The trainer below is nnUNetTrainerSegScratch (lr 1e-3,
300 epochs) with the stock parent. Output lands in
nnunet_results/Dataset{teacher_id}_{name}_CT_7labels/nnUNetTrainerSegScratch__{plans}__{configuration}/fold_{fold},
which is where AFP_from_plans.teacher_paths_from_config looks.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

CORE = Path(__file__).resolve().parent
REPO = CORE.parent


DIR_ENV = {
    "nnUNet_raw": "raw",
    "nnUNet_preprocessed": "preprocessed",
    "nnUNet_results": "results",
}


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return cfg


def set_nnunet_paths() -> None:
    for env_name, default in DIR_ENV.items():
        raw = str(os.environ.get(env_name) or default).strip()
        path = Path(raw)
        if not path.is_absolute():
            path = REPO / path
        os.environ[env_name] = str(path.resolve())
    # torch.compile needs a C compiler. WSL images often lack gcc.
    os.environ.setdefault("nnUNet_compile", "false")


def drop_repo_from_import_path() -> None:
    """`python core/train_teacher.py` from the repo root puts cwd on sys.path.

    That would import this translation fork instead of totalseg's stock nnunetv2.
    """
    repo = REPO.resolve()
    kept: list[str] = []
    for p in sys.path:
        if p in ("", "."):
            continue
        try:
            if Path(p).resolve() == repo:
                continue
        except OSError:
            pass
        kept.append(p)
    sys.path[:] = kept


def check_stock_nnunet() -> None:
    import nnunetv2

    pkg = Path(nnunetv2.__file__).resolve()
    if REPO in pkg.parents and "site-packages" not in pkg.parts:
        raise SystemExit(
            f"imported this repository's nnunetv2 ({pkg}). Run with the TotalSegmentator environment: "
            "uv run --project totalseg python core/train_teacher.py"
        )
    try:
        from nnunetv2.training.loss import AFP_from_plans  # noqa: F401
    except ImportError:
        return
    raise SystemExit(f"nnunetv2 at {pkg} is the translation fork, not the stock package")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "config.yaml")
    ap.add_argument("--fold", default="0", help="integer or 'all'")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--plans", default="nnUNetPlans", help="plans identifier")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained-weights", type=Path, help="nnU-Net checkpoint to initialize from")
    ap.add_argument("--c", dest="continue_training", action="store_true", help="continue from checkpoint_latest")
    ap.add_argument("--val", action="store_true", help="only run validation with checkpoint_final")
    ap.add_argument("--val-best", action="store_true", help="validate with checkpoint_best")
    ap.add_argument("--npz", action="store_true", help="export validation softmax")
    ap.add_argument("--disable-checkpointing", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    args = ap.parse_args()

    cfg = load_config(args.config.resolve())
    set_nnunet_paths()
    drop_repo_from_import_path()
    check_stock_nnunet()

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from torch.backends import cudnn

    from nnunetv2.run.run_training import maybe_load_checkpoint
    from nnunetv2.training.dataloading.utils import unpack_dataset
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    class nnUNetTrainerSegScratch(nnUNetTrainer):
        """Scratch segmentation training: same LR/epochs as SegFT, no pretrained init."""

        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device("cuda")):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.initial_lr = args.lr
            self.num_epochs = args.epochs

    name = str(cfg["dataset_name"])
    teacher_id = int(cfg["teacher_dataset_id"])
    dataset = f"Dataset{teacher_id:03d}_{name}_CT_7labels"
    preprocessed = Path(os.environ["nnUNet_preprocessed"]) / dataset
    plans_file = preprocessed / f"{args.plans}.json"
    dataset_json_file = preprocessed / "dataset.json"
    for p in (plans_file, dataset_json_file):
        if not p.is_file():
            raise SystemExit(f"missing {p}. Run core/extract_pseudo_labels.py first.")

    fold = args.fold if args.fold == "all" else int(args.fold)
    if args.continue_training and args.val:
        raise SystemExit("--c and --val cannot be combined")
    if args.val_best and args.disable_checkpointing:
        raise SystemExit("--val-best needs checkpoints")

    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    plans = load_json(str(plans_file))
    ct_id = int(cfg["cbct_dataset_id"]) + 1
    src_fp = Path(os.environ["nnUNet_preprocessed"]) / f"Dataset{ct_id:03d}_{name}_CT" / "dataset_fingerprint.json"
    if src_fp.is_file():
        want = load_json(str(src_fp))["foreground_intensity_properties_per_channel"]["0"]
        got = plans["foreground_intensity_properties_per_channel"]["0"]
        if abs(float(want["mean"]) - float(got["mean"])) > 1e-2 or abs(float(want["std"]) - float(got["std"])) > 1e-2:
            raise SystemExit(
                f"teacher intensity mean/std {got['mean']:.3f}/{got['std']:.3f} != "
                f"Dataset{ct_id:03d} {want['mean']:.3f}/{want['std']:.3f}. "
                "AFP applies the teacher to student CT/sCT. Re-run "
                "core/extract_pseudo_labels.py so the teacher uses that fingerprint."
            )
    plans["continue_training"] = args.continue_training
    data_id = plans["configurations"][args.configuration]["data_identifier"]
    unpack_dataset(str(preprocessed / data_id), unpack_segmentation=True, overwrite_existing=False)
    dataset_json = load_json(str(dataset_json_file))
    trainer = nnUNetTrainerSegScratch(plans=plans, configuration=args.configuration, fold=fold,
                                      dataset_json=dataset_json, device=device)
    if args.disable_checkpointing:
        trainer.disable_checkpointing = True

    maybe_load_checkpoint(trainer, args.continue_training, args.val,
                          str(args.pretrained_weights) if args.pretrained_weights else None)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not args.val:
        trainer.run_training()
    if args.val_best:
        trainer.load_checkpoint(join(trainer.output_folder, "checkpoint_best.pth"))
    trainer.perform_actual_validation(args.npz)
    print(f"teacher checkpoint: {join(trainer.output_folder, 'checkpoint_best.pth')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
