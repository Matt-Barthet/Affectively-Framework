import os

import numpy as np
from tensorboardX import SummaryWriter
import shutil


def backup(log_dir):
    if not os.path.exists("./Tensorboard/Backups"):
        os.mkdir("./Tensorboard/Backups")
    if os.path.exists(log_dir):
        counter = 1
        while True:
            filename = f"./Tensorboard/Backups/{log_dir.split('/')[-1]}_{counter}"
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
        self.best_arousal_reward = 0

    def on_episode_end(self):
        mean_arousal = np.mean(self.environment.arousal_trace)
        mean_arousal = mean_arousal if not np.isnan(mean_arousal) else 0
        self.best_arousal_reward = np.max([self.best_arousal_reward, mean_arousal])
        self.writer.add_scalar('reward/best_r_a', self.best_arousal_reward, self.episode)
        self.writer.add_scalar('reward/best_score', self.environment.best_score, self.episode)
        self.writer.add_scalar('reward/best_r_b', self.environment.best_cumulative_reward, self.episode)
        self.writer.add_scalar('reward/episode_mean_arousal', np.mean(self.environment.arousal_trace), self.episode)
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

    def best_cell_behavior(self):
        return self.archive.bestCell.behavior_reward

    def best_cell_affect(self):
        return self.archive.bestCell.arousal_reward

    def best_cell_lambda(self):
        return self.archive.bestCell.blended_reward

    def on_step(self):
        self.env.writer.add_scalar('archive/archive size', self.size(), self.step_count)
        self.env.writer.add_scalar('best cell/trajectory length', self.best_cell_length(), self.step_count)
        self.env.writer.add_scalar('best cell/behavior reward', self.best_cell_behavior(), self.step_count)
        self.env.writer.add_scalar('best cell/affect reward', self.best_cell_affect(), self.step_count)
        self.env.writer.add_scalar('best cell/blended reward', self.best_cell_lambda(), self.step_count)
        self.step_count += 1
