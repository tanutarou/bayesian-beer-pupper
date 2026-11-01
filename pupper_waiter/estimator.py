"""Estimate the beer in the glass, with an interval, from how hard the standing robot's servos work.

* :class:`FeaturePipeline` low-passes the per-frame features (servo sag and load), which are noisy because the
  servos chatter around their torque limit.
* :class:`GaussianProcessRegression` maps the filtered features to a predictive Gaussian over the volume. It
  is calibrated offline on recordings with known volumes (``scripts/sim_probe.py``).
* :class:`VolumeKalman` fuses the frame predictions while the robot stands still.

Consecutive frames share the same calibration error, so averaging many of them would make the interval far
too narrow. The calibration therefore splits the held-out error into a bias shared by a whole weighing and
frame noise (:func:`bias_noise_split`); the filter fuses frames with the noise variance and the reported
interval adds the bias variance.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np


def normal_quantile(p: float) -> float:
    """Inverse of the standard normal CDF, by bisection."""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class FeaturePipeline:
    """Causal first-order low-pass, ``alpha = 1 - exp(-dt / tau)`` per frame."""

    def __init__(self, tau: float = 1.0):
        self.tau = tau
        self.state: np.ndarray | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, features: np.ndarray, dt: float) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if self.state is None:
            self.state = features.copy()
        else:
            self.state += (1.0 - np.exp(-dt / self.tau)) * (features - self.state)
        return self.state


class GaussianProcessRegression:
    """Gaussian-process regression with an RBF kernel on standardised features (Rasmussen & Williams, ch. 2).

    The servo load saturates at the torque limit of a weak servo, so the volume is not a linear function of
    the features. Consecutive frames are near duplicates, so every ``subsample``-th frame is kept. The length
    scale and the noise variance (as a fraction of the target variance) are chosen on a small grid by the log
    marginal likelihood.
    """

    LENGTH_SCALES = (6.0, 12.0)
    NOISE_FRACTIONS = (0.1, 0.3)

    def __init__(self, length_scale: float = 6.0, noise_fraction: float = 0.1):
        self.length_scale = length_scale
        self.noise_fraction = noise_fraction
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.target_mean = 0.0
        self.signal_var = 1.0
        self.noise_var = 1.0
        self.train_x: np.ndarray | None = None
        self.alpha_vec: np.ndarray | None = None
        self.chol: np.ndarray | None = None
        self.log_marginal_likelihood = -np.inf

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return self.signal_var * np.exp(-0.5 * d2 / self.length_scale**2)

    def _fit_fixed(self, xs: np.ndarray, t: np.ndarray) -> None:
        self.train_x = xs
        k = self._kernel(xs, xs) + self.noise_var * np.eye(len(xs))
        self.chol = np.linalg.cholesky(k)
        self.alpha_vec = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, t))
        self._precompute()
        self.log_marginal_likelihood = float(
            -0.5 * t @ self.alpha_vec - np.log(np.diag(self.chol)).sum() - 0.5 * len(t) * np.log(2 * np.pi)
        )

    def fit(self, features: np.ndarray, targets: np.ndarray, subsample: int = 3) -> GaussianProcessRegression:
        """Standardise, subsample, and keep the hyper-parameters with the largest marginal likelihood."""
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-9
        self.target_mean = float(targets.mean())
        xs = ((features - self.feature_mean) / self.feature_std)[::subsample]
        t = (targets - self.target_mean)[::subsample]
        target_var = float(t.var()) + 1e-9
        best = None
        for length_scale in self.LENGTH_SCALES:
            for noise_fraction in self.NOISE_FRACTIONS:
                candidate = GaussianProcessRegression(length_scale, noise_fraction)
                candidate.feature_mean, candidate.feature_std = self.feature_mean, self.feature_std
                candidate.target_mean, candidate.signal_var = self.target_mean, target_var
                candidate.noise_var = noise_fraction * target_var
                candidate._fit_fixed(xs, t)
                if best is None or candidate.log_marginal_likelihood > best.log_marginal_likelihood:
                    best = candidate
        self.__dict__.update(best.__dict__)
        return self

    def _precompute(self) -> None:
        # Inverse of the kernel matrix: a prediction then costs one kernel row and a matrix-vector product.
        eye = np.eye(len(self.chol))
        self.k_inv = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, eye))

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and variance (including the noise) for one or many feature vectors."""
        xs = (np.atleast_2d(np.asarray(features, dtype=np.float64)) - self.feature_mean) / self.feature_std
        ks = self._kernel(xs, self.train_x)
        mean = ks @ self.alpha_vec + self.target_mean
        var = self.signal_var - ((ks @ self.k_inv) * ks).sum(axis=1) + self.noise_var
        return mean, np.maximum(var, 1e-9)

    def in_calibration_range(self, var: np.ndarray, fraction: float = 0.8) -> np.ndarray:
        """False where the predictive variance is back at ``fraction`` of the prior variance: the features lie
        outside the calibration set and the prediction is only the prior mean."""
        return np.asarray(var) < fraction * (self.signal_var + self.noise_var)

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean,
            "signal_var": self.signal_var,
            "noise_var": self.noise_var,
            "length_scale": self.length_scale,
            "noise_fraction": self.noise_fraction,
            "train_x": self.train_x,
            "alpha_vec": self.alpha_vec,
            "chol": self.chol,
        }

    @classmethod
    def from_state(cls, d) -> GaussianProcessRegression:
        model = cls(float(d["length_scale"]), float(d["noise_fraction"]))
        model.feature_mean, model.feature_std = d["feature_mean"], d["feature_std"]
        model.target_mean = float(d["target_mean"])
        model.signal_var, model.noise_var = float(d["signal_var"]), float(d["noise_var"])
        model.train_x, model.alpha_vec, model.chol = d["train_x"], d["alpha_vec"], d["chol"]
        model._precompute()
        return model


class VolumeKalman:
    """One-dimensional Kalman filter on the volume with a random-walk transition.

    The prior is deliberately broad. ``gate_ang_vel`` [rad/s] suspends the correction while the body rocks.
    ``bias_var`` is the variance of the calibration error that every frame of a weighing shares: it is added to
    the reported spread (``std``, ``interval``) but not to the filter state.
    """

    def __init__(
        self,
        bias_var: float = 0.0,
        prior_mean: float = 150.0,
        prior_std: float = 150.0,
        process_std_per_s: float = 5.0,
        gate_ang_vel: float = 0.3,
    ):
        self.mean = prior_mean
        self.var = prior_std**2
        self.bias_var = bias_var
        self.process_var_per_s = process_std_per_s**2
        self.gate_ang_vel = gate_ang_vel

    def update(self, measurement: float, measurement_var: float, dt: float, ang_vel_norm: float = 0.0) -> None:
        """Predict over ``dt`` seconds, then correct unless the body is rocking."""
        self.var += self.process_var_per_s * dt
        if ang_vel_norm > self.gate_ang_vel:
            return
        gain = self.var / (self.var + measurement_var)
        self.mean += gain * (measurement - self.mean)
        self.var *= 1.0 - gain

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + self.bias_var))

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """Central interval at the given probability."""
        z = normal_quantile(0.5 + level / 2.0)
        return self.mean - z * self.std, self.mean + z * self.std

    def upper(self, level: float = 0.9) -> float:
        """One-sided upper quantile, e.g. 0.9 -> the value the volume stays below with 90% probability."""
        return self.mean + normal_quantile(level) * self.std


def save_estimator(
    path: str, model: GaussianProcessRegression, blocks: tuple[str, ...], bias_var: float, noise_var: float
) -> None:
    np.savez(path, **model.state_dict(), blocks=np.array(blocks), bias_var_ml2=bias_var, noise_var_ml2=noise_var)


def load_estimator(path: str) -> tuple[GaussianProcessRegression, tuple[str, ...], float, float]:
    """(model, feature blocks, bias variance, frame-noise variance) as written by :func:`save_estimator`."""
    with np.load(path) as d:
        return (
            GaussianProcessRegression.from_state(d),
            tuple(str(b) for b in d["blocks"]),
            float(d["bias_var_ml2"]),
            float(d["noise_var_ml2"]),
        )


def leave_one_group_out(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, level: float = 0.9) -> dict:
    """Frame-level check: calibrate on all but one group (volume) at a time and score the held-out frames.

    Returns the mean absolute error, the coverage of the regression's own central interval at ``level`` and
    the mean predictive standard deviation.
    """
    z = normal_quantile(0.5 + level / 2.0)
    errors, covered, stds = [], [], []
    for g in np.unique(groups):
        train, test = groups != g, groups == g
        mean, var = GaussianProcessRegression().fit(features[train], targets[train]).predict(features[test])
        std = np.sqrt(var)
        errors.append(np.abs(mean - targets[test]))
        covered.append(np.abs(mean - targets[test]) <= z * std)
        stds.append(std)
    return {
        "mae": float(np.concatenate(errors).mean()),
        "coverage": float(np.concatenate(covered).mean()),
        "pred_std": float(np.concatenate(stds).mean()),
    }


def bias_noise_split(features: np.ndarray, targets: np.ndarray, n_traj: int) -> tuple[float, float]:
    """Split the leave-one-volume-out error into a per-trajectory bias and per-frame noise.

    Rows are frame-major (``frame * n_traj + trajectory``) as ``sim_probe.py`` records them and ``targets`` is
    the volume of each row. For each held-out volume, the regression fitted on the others predicts its frames;
    the mean error of a trajectory is its bias, the rest is frame noise. Returns the two variances [ml^2].
    """
    traj_volume = np.asarray(targets).reshape(-1, n_traj)[0]
    n_frames = len(targets) // n_traj
    biases, noises = [], []
    for v in np.unique(traj_volume):
        held = traj_volume == v
        train_rows = np.repeat(~held[None], n_frames, axis=0).reshape(-1)
        mean, _ = (
            GaussianProcessRegression().fit(features[train_rows], targets[train_rows]).predict(features[~train_rows])
        )
        error = (mean - targets[~train_rows]).reshape(n_frames, int(held.sum()))
        biases += error.mean(axis=0).tolist()
        noises += (error - error.mean(axis=0)).reshape(-1).tolist()
    return float(np.var(biases)), float(np.var(noises))


def _final_estimates(
    means: np.ndarray, bias_var: float, noise_var: float, dt: float
) -> list[tuple[float, float, float]]:
    """Run the filter over each column (trajectory) of frame predictions; returns (mean, lo, hi) of the end."""
    finals = []
    for column in means.T:
        kf = VolumeKalman(bias_var=bias_var)
        for m in column:
            kf.update(float(m), noise_var, dt)
        finals.append((kf.mean, *kf.interval(0.9)))
    return finals


def _summary(finals: list[tuple[float, float, float]], truths: np.ndarray) -> dict:
    errors = [abs(mean - truth) for (mean, _, _), truth in zip(finals, truths, strict=True)]
    return {
        "kf_mae_ml": float(np.mean(errors)),
        "kf_q90_ml": float(np.quantile(errors, 0.9)),
        "coverage": float(np.mean([lo <= truth <= hi for (_, lo, hi), truth in zip(finals, truths, strict=True)])),
        "half_width_ml": float(np.mean([0.5 * (hi - lo) for _, lo, hi in finals])),
    }


def score_weighings(means: np.ndarray, truths: np.ndarray, bias_var: float, noise_var: float, dt: float) -> dict:
    """Score the filtered estimate at the end of each weighing (``means``: frames x trajectories): MAE, 90%
    quantile of the error, coverage and mean half-width of the 90% interval."""
    return _summary(_final_estimates(means, bias_var, noise_var, dt), truths)


def held_out_weighings(features: np.ndarray, targets: np.ndarray, n_traj: int, dt: float) -> dict:
    """Score the whole method on volumes it was not calibrated on: an outer leave-one-volume-out for the
    regression and, inside it, :func:`bias_noise_split` on the remaining volumes for the interval."""
    traj_volume = np.asarray(targets).reshape(-1, n_traj)[0]
    n_frames = len(targets) // n_traj
    finals, truths = [], []
    for v in np.unique(traj_volume):
        held = traj_volume == v
        n_held = int(held.sum())
        train_rows = np.repeat(~held[None], n_frames, axis=0).reshape(-1)
        model = GaussianProcessRegression().fit(features[train_rows], targets[train_rows])
        bias_var, noise_var = bias_noise_split(features[train_rows], targets[train_rows], n_traj - n_held)
        mean, _ = model.predict(features[~train_rows])
        finals += _final_estimates(mean.reshape(n_frames, n_held), bias_var, noise_var, dt)
        truths += [v] * n_held
    return _summary(finals, np.array(truths))
