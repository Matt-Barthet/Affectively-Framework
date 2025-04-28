import os

import numpy as np
from tensorboardX import SummaryWriter
import shutil


def backup(log_dir):
    if not os.path.exists("../../Tensorboard/Backups"):
        os.mkdir("../../Tensorboard/Backups")
    if os.path.exists(log_dir):
        counter = 1
        while True:
            filename = f"../../Tensorboard/Backups/{log_dir.split('/')[-1]}_{counter}"
            if not os.path.exists(filename):
                shutil.move(log_dir, f"{filename}")
                break
            counter += 1


class TensorBoardCallback:
    def __init__(self, log_dir, environment):
        self.log_dir = log_dir
        self.environment = environment
        backup(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.episode = 0

        self.best_cumulative_rb = 0
        self.best_env_score = 0
        self.best_cumulative_ra = 0
        self.best_cumulative_rl = 0
        self.best_mean_ra, self.best_mean_rb, self.best_mean_rl = 0, 0, 0


    def on_episode_end(self):

        self.best_cumulative_ra = np.max([self.best_cumulative_ra, self.environment.cumulative_ra])
        self.best_cumulative_rb = np.max([self.best_cumulative_rb, self.environment.cumulative_rb])
        self.best_cumulative_rl = np.max([self.best_cumulative_rl, self.environment.cumulative_rl])

        if self.environment.period_ra:
            # Normalize by length of the episodes arousal trace (static size due to fixed periodic reward)
            mean_rl = np.nan_to_num(self.environment.cumulative_rl / len(self.environment.episode_arousal_trace))
            mean_ra = np.nan_to_num(self.environment.cumulative_ra / len(self.environment.episode_arousal_trace))
            mean_rb = np.nan_to_num(self.environment.cumulative_rb / len(self.environment.episode_arousal_trace))
        else:
            # Normalize based on number of rewards assigned (taken by counting changes in env score)
            mean_rl = 0 if self.environment.behavior_ticks == 0 else self.environment.cumulative_rl / self.environment.behavior_ticks
            mean_ra = 0 if self.environment.behavior_ticks == 0 else self.environment.cumulative_ra / self.environment.behavior_ticks
            mean_rb = 0 if self.environment.behavior_ticks == 0 else self.environment.cumulative_rb / self.environment.behavior_ticks

        self.best_env_score = np.max([self.environment.current_score, self.best_env_score])

        self.writer.add_scalar('affect_rewards/best_r_a', self.environment.best_ra, self.episode)
        self.writer.add_scalar('affect_rewards/cumulative_r_a', self.environment.cumulative_ra, self.episode)
        self.writer.add_scalar('affect_rewards/best_cumulative_r_a', self.best_cumulative_ra, self.episode)
        self.writer.add_scalar('affect_rewards/mean_r_a', mean_ra, self.episode)
        self.writer.add_scalar('affect_rewards/episode_mean_arousal', np.mean(self.environment.episode_arousal_trace), self.episode)

        self.writer.add_scalar('behavior_rewards/best_r_b', self.environment.best_rb, self.episode)
        self.writer.add_scalar('behavior_rewards/cumulative_r_b', self.environment.cumulative_rb, self.episode)
        self.writer.add_scalar('behavior_rewards/best_cumulative_r_b', self.best_cumulative_rb, self.episode)
        self.writer.add_scalar('behavior_rewards/mean_r_b', mean_rb, self.episode)

        self.writer.add_scalar('overall_reward/current_env_score', self.environment.current_score, self.episode)
        self.writer.add_scalar('overall_reward/best_env_score', self.best_env_score, self.episode)
        self.writer.add_scalar('overall_reward/best_r_lambda', self.environment.best_rl, self.episode)
        self.writer.add_scalar('overall_reward/cumulative_r_lambda', self.environment.cumulative_rl, self.episode)
        self.writer.add_scalar('overall_reward/best_cumulative_r_lambda', self.best_cumulative_rl, self.episode)
        self.writer.add_scalar('overall_reward/mean_r_lambda', mean_rl, self.episode)

        self.episode += 1
        self.writer.flush()


class TensorboardGoExplore:
    def __init__(self, env, archive):
        self.env = env
        self.step_count = 0
        self.archive = archive

    def size(self):
        return len(self.archive.archive)

    def best_cell_length(self):
        return self.archive.bestCell.get_cell_length()

    def best_cell_lambda(self):
        return self.archive.bestCell.reward

    def on_step(self):
        self.env.writer.add_scalar('archive/archive size', self.size(), self.step_count)
        self.env.writer.add_scalar('archive/archive updates', self.archive.updates, self.step_count)
        self.env.writer.add_scalar('best cell/trajectory length', self.best_cell_length(), self.step_count)
        self.env.writer.add_scalar('best cell/blended reward', self.best_cell_lambda(), self.step_count)
        self.step_count += 100 * 20
