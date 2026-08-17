from __future__ import annotations

import argparse
import faulthandler

from torch.distributed.elastic.multiprocessing.errors import record

from alaya.config.loader import load_config

# NOTE: trainer imports are deliberately deferred into main(): their import chain
# (cv2/decord) loads a bundled zlib that corrupts torchvision's libpng decoder,
# which the da3_infer dispatch path needs for reading the case image.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alaya rollout training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


@record
def main() -> None:
    faulthandler.enable()
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.da3_infer.enabled:
        # Unified entry for the da3 case-demo inference: the same
        # CONFIG_PATH=... train.sh launch dispatches here instead of a trainer.
        import argparse as _ap

        from inference.run import main as da3_main

        d = cfg.da3_infer
        da3_main(_ap.Namespace(
            input=d.input,
            cfg=args.config,
            output_dir=None,          # falls back to cfg.run.output_dir
            rounds=int(d.rounds),
            seed=int(cfg.run.seed) if cfg.run.seed is not None else None,
            compile=str(d.compile),
            flex_attn=bool(d.flex_attn),
            joystick=d.joystick,
            ttc=bool(d.ttc),
            video_crf=int(d.video_crf),
            skill_sec=float(d.skill_sec),
            skill_prompt=d.skill_prompt,
            skill_keep_wrap=bool(d.skill_keep_wrap),
        ))
        return
    from alaya.trainer.dmd_trainer import DmdTrainer
    from alaya.trainer.frame_query_trainer import FrameQueryTrainer
    from alaya.trainer.rollout_trainer import RolloutTrainer

    if cfg.frame_query.enabled:
        trainer_cls = FrameQueryTrainer
    elif cfg.dmd.enabled:
        trainer_cls = DmdTrainer
    else:
        trainer_cls = RolloutTrainer
    trainer = trainer_cls(cfg)
    if args.describe:
        trainer.describe()
        return
    if args.validate_only:
        trainer.setup()
        trainer.validate(trainer.global_step)
        return
    trainer.train()


if __name__ == "__main__":
    main()
