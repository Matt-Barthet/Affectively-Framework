import os

import numpy as np
from tensorboardX import SummaryWriter
import shutil


def backup(log_dir):
    if not os.path.exists("./results/backups"):
        os.mkdir("./results/backups")
    if os.path.exists(log_dir):
        counter = 1
        while True:
            filename = f"./results/backups/{log_dir.split('/')[-1]}_{counter}"
            if not os.path.exists(filename):
                shutil.move(log_dir, f"{filename}")
                break
            counter += 1


import matplotlib.pyplot as plt
from collections import deque
import time

class InteractiveDashboard:

    def __init__(self, environment=None, max_len=1000, figsize=(8, 4)):

        self.environment = environment if environment is not None else globals().get("environment", None)

        self.max_len = max_len
        self.times = deque(maxlen=max_len)
        self.scores = deque(maxlen=max_len)
        self.arousals = deque(maxlen=max_len)

        self.fig = None
        self.ax_score = None
        self.ax_arousal = None
        self.line_score = None
        self.line_arousal = None
        self.cluster_score = None
        self.cluster_arousal = None
        self.last_draw = 0.0
        self.figsize = figsize

        try:
            plt.ion()
        except Exception:
            pass

        self._local_step = 0

    def _get_current_score(self):
        env = self.environment
        if env is None:
            return None

        if hasattr(env, "current_score"):
            return getattr(env, "current_score")

        for attr in ("score", "get_score", "reward"):
            if hasattr(env, attr):
                val = getattr(env, attr)
                return val() if callable(val) else val
        return None

    def _get_current_arousal(self):
        env = self.environment
        if env is None:
            return None

        if hasattr(env, "current_arousal"):
            return getattr(env, "current_arousal")

        for trace_name in ("episode_arousal_trace", "arousal_trace", "arousal_history", "arousal"):
            if hasattr(env, trace_name):
                trace = getattr(env, trace_name)
                try:

                    if len(trace) > 0:
                        return trace[-1]
                except Exception:

                    return trace

        if hasattr(env, "cumulative_ra") and hasattr(env, "behavior_ticks") and env.behavior_ticks:
            return env.cumulative_ra / float(env.behavior_ticks)
        return None

    def _ensure_fig(self):
        if self.fig is not None:
            return

        self.fig, (self.ax_score, self.ax_arousal) = plt.subplots(
            2, 1, figsize=self.figsize, sharex=True, gridspec_kw={"height_ratios": (1, 1.2)}
        )

        try:
            manager = plt.get_current_fig_manager()
            manager.set_window_title("Interactive Dashboard")
        except Exception:
            pass

        self.ax_arousal.set_ylim([0, 1])
        self.ax_score.set_ylabel("Env score")
        self.ax_score.grid(True, linestyle=':', linewidth=0.5)
        self.ax_arousal.set_ylabel("Arousal")
        self.ax_arousal.set_xlabel("Time / steps")
        self.ax_arousal.grid(True, linestyle=':', linewidth=0.5)

        (self.line_score,) = self.ax_score.plot([], [], linestyle='-', marker=None, linewidth=2.0, label="You")
        (self.line_arousal,) = self.ax_arousal.plot([], [], linestyle='-', marker=None, linewidth=2.0, label="You")

        (self.cluster_score, )= self.ax_score.plot([], [], linestyle='-', marker=None, linewidth=2.0, color='orange', label="Experts")
        (self.cluster_arousal, )= self.ax_arousal.plot([], [], linestyle='-', marker=None, linewidth=2.0, color='orange', label="Experts")
        self.ax_score.legend(loc='upper left')
        self.fig.tight_layout()

        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass


    def on_step(self):
        if self.environment is None:
            self.environment = globals().get("environment", None)

        score = self._get_current_score()
        arousal = self._get_current_arousal()

        t = None
        if hasattr(self.environment, "step_count"):
            try:
                t = int(getattr(self.environment, "step_count"))
            except Exception:
                t = None
        if t is None and hasattr(self.environment, "tick"):
            try:
                t = int(getattr(self.environment, "tick"))
            except Exception:
                t = None
        if t is None:
            self._local_step += 1
            t = self._local_step

        self.times.append(t)
        self.scores.append(np.nan if score is None else score)
        self.arousals.append(np.nan if arousal is None else arousal)

        self._ensure_fig()

        xs = list(self.times)
        ys_score = list(self.scores)
        ys_arousal = list(self.arousals)

        try:
            self.line_score.set_data(xs, ys_score)
            self.line_arousal.set_data(xs, ys_arousal)

            self.cluster_score.set_data(xs, self.environment.model.cluster_score[:len(xs)])
            self.cluster_arousal.set_data(xs, self.environment.model.cluster_arousal[:len(xs)])

            if xs:
                xmin, xmax = min(xs), max(xs)
                xpad = max(1, int((xmax - xmin) * 0.02))
                self.ax_score.set_xlim(xmin - xpad, xmax + xpad)
                self.ax_arousal.set_xlim(xmin - xpad, xmax + xpad)

            if any([not np.isnan(v) for v in ys_score]):
                ymin = 0
                ymax = np.nanmax(np.concatenate([ys_score, self.environment.model.cluster_score[:len(xs)]]))
                if ymin == ymax:
                    ymin = 0
                    ymax = 1
                ypad = (ymax - ymin) * 0.05
                self.ax_score.set_ylim(ymin - ypad, ymax + ypad)

            if any([not np.isnan(v) for v in ys_arousal]):
                ymin = 0
                ymax = 1
                ypad = (ymax - ymin) * 0.05
                self.ax_arousal.set_ylim(ymin - ypad, ymax + ypad)

            now = time.time()
            self.fig.canvas.draw()
            try:
                self.fig.canvas.flush_events()
            except Exception:
                plt.pause(0.001)
            self.last_draw = now
        except Exception:
            try:
                self.fig = None
                self._ensure_fig()
            except Exception:
                pass



class TensorBoardCallback:

    def __init__(self, log_dir, environment, model):
        self.log_dir = log_dir
        self.environment = environment
        backup(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.episode = 0
        self.model = model
        self.best_cumulative_rb = 0
        self.best_env_score = 0
        self.best_cumulative_ra = 0
        self.best_cumulative_rl = 0
        self.best_mean_ra, self.best_mean_rb, self.best_mean_rl = 0, 0, 0


    def on_episode_end(self):

        self.best_cumulative_ra = np.max([self.best_cumulative_ra, self.environment.cumulative_ra])
        self.best_cumulative_rb = np.max([self.best_cumulative_rb, self.environment.cumulative_rb])
        self.best_cumulative_rl = np.max([self.best_cumulative_rl, self.environment.cumulative_rl])
        self.best_env_score = np.max([self.environment.current_score, self.best_env_score])

        print(f"End of episode, Best Cumulative Rb =  {self.best_cumulative_rb}")

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

        self.episode += 1

        # if self.episode % 100 == 0:

        # Arousal metrics
        self.writer.add_scalar('affect_rewards/cumulative_r_a', self.environment.cumulative_ra, self.episode)
        self.writer.add_scalar('affect_rewards/best_cumulative_r_a', self.best_cumulative_ra, self.episode)
        self.writer.add_scalar('affect_rewards/mean_r_a', mean_ra, self.episode)
        self.writer.add_scalar('affect_rewards/episode_mean_arousal', np.mean(self.environment.episode_arousal_trace), self.episode)

        # Behavior Metrics
        self.writer.add_scalar('behavior_rewards/cumulative_r_b', self.environment.cumulative_rb, self.episode)
        self.writer.add_scalar('behavior_rewards/best_cumulative_r_b', self.best_cumulative_rb, self.episode)
        self.writer.add_scalar('behavior_rewards/mean_r_b', mean_rb, self.episode)

        # General Metrics
        self.writer.add_scalar('overall_reward/current_env_score', self.environment.current_score, self.episode)
        self.writer.add_scalar('overall_reward/best_env_score', self.best_env_score, self.episode)
        self.writer.add_scalar('overall_reward/cumulative_r_lambda', self.environment.cumulative_rl, self.episode)
        self.writer.add_scalar('overall_reward/best_cumulative_r_lambda', self.best_cumulative_rl, self.episode)
        self.writer.add_scalar('overall_reward/mean_r_lambda', mean_rl, self.episode)

        if self.episode % 1000 == 0:
            self.model.save(f"{self.log_dir}-Episode-{self.episode}.zip")

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
