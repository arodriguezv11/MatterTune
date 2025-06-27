from __future__ import annotations

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
from pathlib import Path

import rich

import mattertune.configs as MC
from mattertune import MatterTuner


def hparams():
    hparams = MC.MatterTunerConfig.draft()

    # ---------------------
    # Dataset Configuration
    # ---------------------
    hparams.data = MC.ManualSplitDataModuleConfig(
        train=MC.JSONDatasetConfig(
            src=Path("examples/descriptors/datasets/descriptor_pretrain.json"),
            tasks={
                "GMP": "GMP"  # map JSON field to internal task
            },
        ),
        validation=None,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
    )

    # ---------------------
    # Model Configuration
    # ---------------------
    hparams.model = MC.ORBBackboneConfig(
        name="orb",
        pretrained_model="orb-v2",  # or orb-v3 etc.
        reset_backbone=False,  # Keep backbone weights
        freeze_backbone=False,  # Allow updates
        reset_output_heads=True,  # Reset heads so only 'descriptor_gmp' is supervised
        use_pretrained_normalizers=False,
        ignore_gpu_batch_transform_error=True,
        normalizers={},  # No energy/force/stress norm
        optimizer=MC.AdamWConfig(
            name="AdamW",
            lr=1e-4,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            amsgrad=False,
        ),
        system=MC.ORBSystemConfig(radius=6.0, max_num_neighbors=120),
        properties=[
            MC.GMPPropertyConfig(
                name="GMP",
                dtype="float",
                loss=MC.MSELossConfig(name="mse", reduction="mean"),
                loss_coefficient=1.0,
                conservative=False,
            )
        ],
    )

    # ---------------------
    # Trainer Configuration
    # ---------------------
    hparams.trainer = MC.TrainerConfig(
        accelerator="gpu",
        strategy="auto",
        devices=[0],
        precision="32-true",
        max_epochs=100,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        additional_trainer_kwargs={"inference_mode": False},
        loggers="default",
    )

    hparams = hparams.finalize(strict=False)
    rich.print(hparams)
    return hparams


if __name__ == "__main__":
    config = hparams()
    print("Loaded properties:")
    for prop in config.model.properties:
        print(
            f"  - {prop.name}, conservative = {getattr(prop, 'conservative', 'MISSING')}"
        )

    tuner = MatterTuner(config)
    model, trainer = tuner.tune()

    # Save checkpoint
    trainer.save_checkpoint("checkpoints/orb_descriptor_pretrained.ckpt")
