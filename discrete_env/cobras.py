from typing import Optional
import math
import numpy as np

from discrete_env.helper_pre_vec import StartSpace
from discrete_env.pre_vec_env import PreVecEnv, create_pre_vec


class CobraEnv(PreVecEnv):
    def __init__(self, n_envs,
                 n_cobras_start_low=2,
                 n_cobras_start_high=10,
                 max_cobras=100,
                 max_heads=100,
                 max_steps=500,
                 seed=0,
                 render_mode: Optional[str] = None, ):
        self.n_envs = n_envs
        self.n_cobras_start_low = n_cobras_start_low
        self.n_cobras_start_high = n_cobras_start_high
        self.max_cobras = max_cobras
        self.max_heads = max_heads
        n_actions = 2
        # action 0 -> breed
        self.breed = 0
        # action 1 -> decapitate
        self.decap = 1
        self.n_features = 2
        # feature 0 -> n_cobras
        # feature 1 -> n_cobra_heads
        # self.state = np.zeros((self.n_envs, self.n_features))
        self.low = np.array([0, 0], dtype=np.float32)
        self.high = np.array([self.max_cobras, self.max_heads], dtype=np.float32)

        self.start_space = StartSpace(
            low=[self.n_cobras_start_low, 0],
            high=[self.n_cobras_start_high, 0],
            np_random=self._np_random,
            discrete_space=True,
        )

        self.reward = np.full(n_envs, 0.0)
        self.info = [{"env_reward": self.reward[i]} for i in range(n_envs)]

        self.customizable_params = [
            "n_cobras_start_low",
            "n_cobras_start_high",
            "max_cobras",
            "max_heads",
            "max_steps",
        ]

        super().__init__(n_envs, n_actions, "Cobras", max_steps, seed, render_mode)

    def get_ob_names(self):
        return [
            "Cobra Population",
            "Cobra Heads",
        ]

    def transition_model(self, action: np.array):
        n_cobras, n_heads = self.state.T

        breed = np.bitwise_and(n_cobras > 1, action == self.breed)
        decapitate = np.bitwise_and(n_cobras > 0, action == self.decap)
        n_cobras[breed] *= 2
        n_cobras[decapitate] -= 1
        n_heads[decapitate] += 1

        self.terminated = np.bitwise_or(n_cobras >= self.max_cobras, n_heads >= self.max_heads)

        self.state = np.vstack((n_cobras, n_heads)).T

        self.reward = n_heads
        self.info = [{"env_reward": self.reward[i]} for i in range(self.num_envs)]

    def get_action_lookup(self):
        return {
            0: 'Breed',
            1: 'Decapitate',
        }

    def render_unique(self):
        """
        Vector-friendly renderer for the Cobras environment.

        Each environment is shown in its own cell of a regular grid.
        Within a cell we draw two vertical bars:

            • left  – number of cobras        (black)
            • right – number of cobra heads   (dark grey)

        Above each bar we print the raw integer.  Heights are proportional to
        the respective maxima so every cell is on the same scale.
        """
        # ─── Unpack vector state ────────────────────────────────────────────────
        n_cobras, n_heads = self.state.T  # shape: (self.n_envs,)
        n_envs = self.n_envs
        sc_w, sc_h = self.screen_width, self.screen_height

        # ─── Create fresh surface ───────────────────────────────────────────────
        self.surf = self.pygame.Surface((sc_w, sc_h))
        self.surf.fill((255, 255, 255))

        # ─── Grid geometry ──────────────────────────────────────────────────────
        cols = math.ceil(math.sqrt(n_envs))  # roughly square grid
        rows = math.ceil(n_envs / cols)
        cell_w = sc_w / cols
        cell_h = sc_h / rows
        margin_x = cell_w * 0.15  # breathing-space inside a cell
        bar_w = (cell_w - 3 * margin_x) / 2  # two bars plus three gaps

        # ─── Font for numbers (auto-scaled) ─────────────────────────────────────
        font_size = max(12, int(cell_h * 0.18))
        font = self.pygame.font.SysFont(None, font_size)

        # ─── Pre-compute scaling for heights ────────────────────────────────────
        h_max_cobras = max(1, self.max_cobras)
        h_max_heads = max(1, self.max_heads)
        usable_h = cell_h * 0.60  # leave space for labels

        # ─── Iterate through every environment ──────────────────────────────────
        for idx in range(n_envs):
            row, col = divmod(idx, cols)
            x0 = col * cell_w
            # x0 = (cols - col - 1) * cell_w

            y0 = row * cell_h

            # Cell outline (light grey)
            self.pygame.draw.rect(
                self.surf, (200, 200, 200),
                self.pygame.Rect(int(x0), int(y0), int(cell_w), int(cell_h)), 1
            )

            # Data for this env
            c_count = int(n_cobras[idx])
            h_count = int(n_heads[idx])

            # Heights (0 → bottom, max → usable_h)
            c_h = (c_count / h_max_cobras) * usable_h
            h_h = (h_count / h_max_heads) * usable_h

            # Bar top-left coordinates
            c_x = x0 + margin_x
            h_x = c_x + bar_w + margin_x
            bar_base = y0 + cell_h * 0.90  # a bit above the bottom

            # Draw cobra-count bar (solid black)
            c_rect = self.pygame.Rect(
                int(c_x), int(bar_base - c_h),
                int(bar_w), int(c_h)
            )
            self.pygame.draw.rect(self.surf, (0, 0, 0), c_rect)

            # Draw head-count bar (dark grey)
            h_rect = self.pygame.Rect(
                int(h_x), int(bar_base - h_h),
                int(bar_w), int(h_h)
            )
            self.pygame.draw.rect(self.surf, (96, 96, 96), h_rect)

            # Integer labels centred above each bar
            for text, centre_x in (
                    (str(c_count), c_x + bar_w / 2),
                    (str(h_count), h_x + bar_w / 2),
            ):
                rendered = font.render(text, True, (0, 0, 0))
                t_rect = rendered.get_rect(midbottom=(centre_x, y0 + cell_h * 0.25))
                self.surf.blit(rendered, t_rect)

            # Short legend beneath the bars (optional clarity)
            label_font = self.pygame.font.SysFont(None, max(10, int(cell_h * 0.13)))
            for label, centre_x in (
                    ("Cobras", c_x + bar_w / 2),
                    ("Heads", h_x + bar_w / 2),
            ):
                lbl = label_font.render(label, True, (0, 0, 0))
                l_rect = lbl.get_rect(midtop=(centre_x, y0 + cell_h * 0.93))
                self.surf.blit(lbl, l_rect)
        # Un-mirror the whole surface so text isn’t reversed
        self.surf = self.pygame.transform.flip(self.surf, False, True)
        return True


def create_cobras(args, hyperparameters, is_valid=False):
    param_range = {
        "n_cobras_start_low": [2],
        "n_cobras_start_high": [10],
        "max_cobras": [100],
        "max_heads": [100],
    }
    return create_pre_vec(args, hyperparameters, param_range, CobraEnv, is_valid)
