import os
from tensorboardX import SummaryWriter
import shutil
import sys
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from morl_baselines.common.performance_indicators import hypervolume


class HypervolumeTracker:
    def __init__(self, reference_point):
        self.reference_point = np.asarray(reference_point, dtype=np.float32)

    def compute(self, pareto_front):
        if pareto_front is None or len(pareto_front) == 0:
            return 0.0
        return hypervolume(
            np.asarray(pareto_front, dtype=np.float32),
            self.reference_point
        )

class MORLTensorBoardCallback:
    def __init__(self, log_dir, environment, agent, reference_point):
        self.log_dir = log_dir
        self.environment = environment
        backup(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.agent = agent
        self.episode = 0

        self.best_cumulative_ra = 0
        self.best_cumulative_rb = 0
        self.best_env_score = 0
        self.best_hypervolume = 0

        self.hv_tracker = HypervolumeTracker(reference_point)

    def on_episode_end(self):

        env = self.environment
        self.episode += 1

        r_a = env.env.cumulative_ra
        r_b = env.env.cumulative_rb

        self.best_cumulative_ra = max(self.best_cumulative_ra, r_a)
        self.best_cumulative_rb = max(self.best_cumulative_rb, r_b)
        self.best_env_score = max(self.best_env_score, env.env.current_score)

        # --- logging ---
        self.writer.add_scalar("returns/r_a", r_a, self.episode)
        self.writer.add_scalar("returns/r_b", r_b, self.episode)

        self.writer.add_scalar("returns/best_r_a", self.best_cumulative_ra, self.episode)
        self.writer.add_scalar("returns/best_r_b", self.best_cumulative_rb, self.episode)

        self.writer.add_scalar("env/current_score", env.env.current_score, self.episode)
        self.writer.add_scalar("env/best_score", self.best_env_score, self.episode)

        # --- arousal diagnostics ---
        mean_arousal = 0.0 if len(env.env.episode_arousal_trace) == 0 else np.mean(env.env.episode_arousal_trace)
        self.writer.add_scalar("affect/episode_mean_arousal", mean_arousal, self.episode)

        if self.episode % 1000 == 0:
            self.agent.save(f"{self.log_dir}-Episode-{self.episode}")

        self.writer.flush()

    def on_step(self):
        pass



class InteractiveDashboard(QtWidgets.QMainWindow):
    """
    Drop-in replacement for your Matplotlib InteractiveDashboard.
    Same API, same on_step(), but rendered in a FAST PyQtGraph window.
    """

    _app = None  # One global QApplication for all dashboards

    def __init__(self, environment=None, max_len=1000, update_every=5):
        # Ensure one QApplication exists
        if InteractiveDashboard._app is None:
            InteractiveDashboard._app = QtWidgets.QApplication(sys.argv)

        super().__init__()

        self.environment = environment
        self.max_len = max_len
        self.update_every = update_every  # Update display every N steps
        self.step_counter = 0

        self.times = deque(maxlen=max_len)
        self.scores = deque(maxlen=max_len)
        self.arousals = deque(maxlen=max_len)
        self._local_step = 0
        self._waiting_restart = False
        self.pause = False
        self._end_label = None

        # ---------- Window ----------
        self.setWindowTitle("Interactive RL Dashboard")
        self.resize(1280, 500)

        cw = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(cw)
        self.setCentralWidget(cw)

        pg.setConfigOptions(antialias=True)
        font = pg.QtGui.QFont("Arial", 16)

        # ---------- Score Plot ----------
        self.score_plot = pg.PlotWidget(title="Live Score Trace")
        self.score_plot.setTitle("Live Score Trace", size='18pt')
        self.score_plot.setLabel(
            'left',
            '<span style="color:gray; font-size:18pt;">Score</span>'
        )
        self.score_plot.getAxis('left').setStyle(tickFont=font)
        self.score_plot.getAxis('bottom').setStyle(tickFont=font)
        layout.addWidget(self.score_plot)

        self.score_legend = self.score_plot.addLegend(
            offset=(10, 10),
            anchor=(0, 0) ,
            colCount=2
        )
        self.score_legend.setLabelTextSize('16pt')

        self.score_curve = self.score_plot.plot([], [], pen=pg.mkPen('cyan', width=2), name="You")
        self.score_expert_curve = self.score_plot.plot([], [], pen=pg.mkPen('orange', width=2), name="Experts")

        # ---------- Arousal Plot ----------

        self.arousal_plot = pg.PlotWidget(title="Live Arousal Trace")
        self.arousal_plot.setTitle("Live Arousal Trace", size='18pt')
        self.arousal_plot.setYRange(0, 1)

        # Correct: use font=QFont(...)
        # Axis tick label fonts
        self.arousal_plot.getAxis('left').setStyle(tickFont=font)
        self.arousal_plot.getAxis('bottom').setStyle(tickFont=font)
        self.arousal_plot.setLabel(
            'left',
            '<span style="color:gray; font-size:18pt;">Arousal</span>'
        )
        self.arousal_plot.setLabel(
            'bottom',
            '<span style="color:gray; font-size:18pt;">Time (s)</span>'
        )
        layout.addWidget(self.arousal_plot)

        # Add legend to arousal plot
        # self.arousal_legend = self.arousal_plot.addLegend()

        self.arousal_curve = self.arousal_plot.plot([], [], pen=pg.mkPen('cyan', width=2), name="You")
        self.arousal_expert_curve = self.arousal_plot.plot([], [], pen=pg.mkPen('orange', width=2), name="Experts")

        self.show()

        # Process events once to show the window
        QtWidgets.QApplication.processEvents()
        self.on_episode_end()


    def on_pause(self):
        """
        Called when the environment is paused. Pauses updating the plots.
        """
        self.pause = True

        # Add a prominent label to the score plot (centered)
        try:
            if self._end_label is None:
                self._end_label = pg.TextItem(
                    html='<div style="color:yellow; font-size:18pt; font-weight:bold">Episode Paused — Press Pause to Continue</div>',
                    anchor=(0.5, 0.5)
                )
                vb = self.score_plot.getViewBox()
                rect = vb.viewRect()
                cx = (rect.left() + rect.right()) / 2.0
                cy = (rect.top() + rect.bottom()) / 2.0
                self._end_label.setPos(cx, cy)
                self.score_plot.addItem(self._end_label)
        except Exception:
            pass

    # -------------------------------------------------------------
    # IDENTICAL API TO YOUR ORIGINAL DASHBOARD
    # -------------------------------------------------------------
    def on_step(self):

        if self._waiting_restart:
            return

        if self.pause:
            return

        if self._end_label is not None:
            self.score_plot.removeItem(self._end_label)
            self._end_label = None

        """
        Called every step in your environment, just like the Matplotlib version.
        """
        env = self.environment

        # Determine t like your original code
        if hasattr(env, "step_count"):
            t = int(env.step_count)
        elif hasattr(env, "tick"):
            t = int(env.tick)
        else:
            self._local_step += 1
            t = self._local_step

        # Score
        score = self._get_score(env)
        self.times.append(t)
        self.scores.append(score if score is not None else np.nan)

        # Arousal
        ar = self._get_arousal(env)
        self.arousals.append(ar if ar is not None else np.nan)

        # Update display periodically
        self.step_counter += 1
        if self.step_counter >= self.update_every:
            self._update_graph()
            QtWidgets.QApplication.processEvents()
            self.step_counter = 0


    # ---------- Score & Arousal helpers ----------
    def _get_score(self, env):
        if hasattr(env, "current_score"):
            return env.current_score

        for attr in ("score", "get_score", "reward"):
            if hasattr(env, attr):
                val = getattr(env, attr)
                return val() if callable(val) else val
        return None

    def _get_arousal(self, env):
        if hasattr(env, "current_arousal"):
            return env.current_arousal

        for n in ("episode_arousal_trace", "arousal_trace", "arousal_history", "arousal"):
            if hasattr(env, n):
                trace = getattr(env, n)
                try:
                    return trace[-1] if len(trace) else None
                except:
                    return trace

        if hasattr(env, "cumulative_ra") and hasattr(env, "behavior_ticks") and env.behavior_ticks:
            return env.cumulative_ra / float(env.behavior_ticks)

        return None

    # -------------------------------------------------------------
    # Redraw plots (fast!)
    # -------------------------------------------------------------
    def _update_graph(self):
        if not self.times:
            return

        xs = np.array(self.times) / 5

        ymin, ymax = float(np.array(self.scores).min()), float(np.array(self.scores).max())

        if ymin == ymax:
            self.score_plot.setYRange(0, 1)
        else:
            self.score_plot.setYRange(ymin, ymax)

        self.score_plot.getViewBox().update()      # update viewbox

        # "You"
        self.score_curve.setData(xs, np.array(self.scores))
        self.arousal_curve.setData(xs, np.array(self.arousals))

        # Experts (if available)
        try:
            m = self.environment.model
            self.score_expert_curve.setData(xs, m.cluster_score[:len(xs)])
            ymin, ymax = min(ymin, float(m.cluster_score[:len(xs)].min())), max(ymax, float(m.cluster_score[:len(xs)].max()))
            self.arousal_expert_curve.setData(xs, m.cluster_arousal[:len(xs)])
        except:
            pass

        if ymin == ymax:
            self.score_plot.setYRange(0, 1)
        else:
            self.score_plot.setYRange(ymin, ymax)

    def on_episode_end(self):
        """
        Called when an episode ends. Clears the current traces and displays
        an overlay label instructing the user to "Press Pause to Restart".

        The dashboard will pause updating plots until it detects the
        environment has restarted (step/tick resets or env.paused toggles).
        """

        # Add a prominent label to the score plot (centered)
        try:
            if self._end_label is None:
                self._end_label = pg.TextItem(
                    html='<div style="color:yellow; font-size:18pt; font-weight:bold">Episode ended — Press Pause to Start</div>',
                    anchor=(0.5, 0.5)
                )
                vb = self.score_plot.getViewBox()
                rect = vb.viewRect()
                cx = (rect.left() + rect.right()) / 2.0
                cy = (rect.top() + rect.bottom()) / 2.0
                self._end_label.setPos(cx, cy)
                self.score_plot.addItem(self._end_label)
        except Exception:
            raise

        # Enter waiting state until a restart/pause signal is detected
        self._waiting_restart = True

    def clear(self):
        try:
            self.times.clear()
            self.scores.clear()
            self.arousals.clear()
        except Exception:
            self.times = deque(maxlen=self.max_len)
            self.scores = deque(maxlen=self.max_len)
            self.arousals = deque(maxlen=self.max_len)

        # Clear plotted data
        try:
            self.score_curve.setData([], [])
            self.arousal_curve.setData([], [])
            self.score_expert_curve.setData([], [])
            self.arousal_expert_curve.setData([], [])
        except Exception:
            pass


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


class TensorBoardCallback:

    def __init__(self, log_dir, environment, model):
        self.log_dir = log_dir
        self.environment = environment.env
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

        # print(f"End of episode, Best Cumulative Rb =  {self.best_cumulative_rb}")

        if self.environment.period_ra:
            # Normalize by length of the episodes arousal trace (static size due to fixed periodic reward)
            trace_len = len(self.environment.episode_arousal_trace)
            mean_rl = 0 if trace_len == 0 else self.environment.cumulative_rl / trace_len
            mean_ra = 0 if trace_len == 0 else self.environment.cumulative_ra / trace_len
            mean_rb = 0 if trace_len == 0 else self.environment.cumulative_rb / trace_len
        else:
            # Normalize based on number of rewards assigned (taken by counting changes in env score)
            mean_rl = 0 if self.environment.behavior_ticks == 0 else self.environment.cumulative_rl / self.environment.behavior_ticks
            mean_ra = 0 if self.environment.behavior_ticks == 0 else self.environment.cumulative_ra / self.environment.behavior_ticks
            mean_rb = 0 if self.environment.behavior_ticks == 0 else self.environment.cumulative_rb / self.environment.behavior_ticks


        self.best_mean_ra = np.max([self.best_mean_ra, mean_ra])
        self.best_mean_rb = np.max([self.best_mean_rb, mean_rb])
        
        self.episode += 1

        # Arousal metrics
        self.writer.add_scalar('affect_rewards/cumulative_r_a', self.environment.cumulative_ra, self.episode)
        self.writer.add_scalar('affect_rewards/best_cumulative_r_a', self.best_cumulative_ra, self.episode)
        self.writer.add_scalar('affect_rewards/mean_r_a', mean_ra, self.episode)

        # Handle empty arousal trace
        episode_mean_arousal = 0 if len(self.environment.episode_arousal_trace) == 0 else np.mean(
            self.environment.episode_arousal_trace)
        self.writer.add_scalar('affect_rewards/episode_mean_arousal', episode_mean_arousal, self.episode)

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

        if self.episode % 1000 == 0 and self.log_dir != "":
            self.model.save(f"{self.log_dir}-Episode-{self.episode}.zip")

        self.writer.flush()

    def on_step(self):
        pass


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