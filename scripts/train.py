import sys

import pyrallis
from diffusers.utils import check_min_version

sys.path.append(".")
sys.path.append("..")

from training.coach import Coach
from training.config import RunConfig
import pidfile

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")


@pyrallis.wrap()
def main(cfg: RunConfig):
    print(f"RUNNING WITH {cfg.data.train_data_dir}")
    prepare_directories(cfg=cfg)
    print("DIR ", cfg.log.exp_dir)
    pidfile.exit_if_job_done(cfg.log.exp_dir)
    coach = Coach(cfg)
    coach.train()
    pidfile.mark_job_done(cfg.log.exp_dir)


def prepare_directories(cfg: RunConfig):
    cfg.log.exp_dir = cfg.log.exp_dir / cfg.log.exp_name
    cfg.log.exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.log.logging_dir = cfg.log.exp_dir / cfg.log.logging_dir
    cfg.log.logging_dir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main()
