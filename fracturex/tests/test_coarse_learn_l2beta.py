"""Unit tests for D13 L2-beta: spectral labels, amplitude model, training.

Covers:
  - ideal_interface_amplitude: in [0,1], large on high-contrast interface nodes;
  - two-level kappa: enrichment with a good mode REDUCES kappa on a synthetic
    high-contrast Schur block (the spectral mechanism, decoupled from the FE solver);
  - amplitude model: bounded output, standardization baked in, numpy inference;
  - training smoke: loss decreases on a tiny split; isolation (no solver import).

Run:
  PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest fracturex/tests/test_coarse_learn_l2beta.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from fracturex.ml.coarse_features import FEATURE_NAMES, N_FEATURES
from fracturex.ml.spectral_labels import (
    ideal_interface_amplitude,
    make_two_level_minv,
    two_level_kappa_dense,
)

torch = pytest.importorskip("torch")


def _feat_block(nn=40):
    """Synthetic features: a localized interface band (high contrast) + smooth bulk."""
    phi = np.zeros((nn, N_FEATURES))
    band = np.arange(nn) >= nn - 8
    phi[:, FEATURE_NAMES.index("d")] = np.linspace(0, 1, nn)
    phi[band, FEATURE_NAMES.index("gradd_l0")] = 0.4
    phi[band, FEATURE_NAMES.index("log_g_over_gbar")] = -5.0  # big contrast on band
    return phi, band


# ---------------------------------------------------------------------------
# spectral labels
# ---------------------------------------------------------------------------

def test_ideal_amplitude_range_and_localization():
    phi, band = _feat_block()
    a = ideal_interface_amplitude(phi)
    assert a.shape == (phi.shape[0],)
    assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-12
    # band (high contrast + steep grad) should carry larger amplitude than the bulk.
    assert a[band].mean() > a[~band].mean() + 0.3


def _p1_prolongation(nc):
    """Continuous linear P1 prolongation, fine ``n=2nc-1`` <- coarse ``nc``."""
    n = 2 * nc - 1
    rows, cols, vals = [], [], []
    for i in range(n):
        c = i / 2.0
        c0 = int(np.floor(c)); c1 = min(c0 + 1, nc - 1); w = c - c0
        rows += [i, i]; cols += [c0, c1]; vals += [1 - w, w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, nc)), n


def _highcontrast_chain(nc=12, contrast=1e3):
    """High-contrast 1D SPD chain + P1 prolongation (a localized stiff coefficient).

    Returns ``(S, PI)``. A stiff coefficient on the central edges creates a sharp
    interface whose worst-conditioned two-level mode is outside range(PI_s) -- the
    plan-command-0 setting in miniature.
    """
    PI, n = _p1_prolongation(nc)
    coef = np.ones(n - 1)
    coef[n // 2 - 1: n // 2 + 1] = contrast  # stiff edges at the interface
    main = np.zeros(n)
    off = -coef
    main[1:] += coef
    main[:-1] += coef
    S = sp.diags([off, main, off], [-1, 0, 1]).tocsr() + 1e-8 * sp.eye(n)
    return S.tocsr(), PI


def _worst_mode(S, Minv):
    """Generalized eigenvector of ``M^{-1} S`` at the extreme (min) eigenvalue."""
    n = S.shape[0]
    Sd = S.toarray()
    T = np.column_stack([Minv.matvec(Sd[:, j]) for j in range(n)])
    T = 0.5 * (T + T.T)
    ev, evec = np.linalg.eigh(T)
    ev_pos = ev[ev > 1e-12 * ev.max()]
    # the mode driving kappa is the smallest positive eigenvalue's eigenvector
    idx = int(np.argmin(np.where(ev > 1e-12 * ev.max(), ev, np.inf)))
    return evec[:, idx], float(ev_pos.max() / ev_pos.min())


def test_two_level_kappa_improves_with_worst_mode():
    """Enriching with the spectrally-worst mode reduces the two-level kappa.

    Honest mechanism test (plan command 0/6): the worst-conditioned direction of the
    baseline two-level operator is outside what coarse+smoother handles; adding it as an
    enrichment column lifts the offending eigenvalue and lowers kappa. This is exactly
    what spectral training (target A1) approximates by learning amplitudes.
    """
    S, PI = _highcontrast_chain(contrast=1e3)
    M0 = make_two_level_minv(S, PI)
    vstar, k0 = _worst_mode(S, M0)
    W = vstar.reshape(-1, 1)
    pinv = np.linalg.pinv(W.T @ (S @ W))
    k1 = two_level_kappa_dense(S, make_two_level_minv(S, PI, enrich_W=W, enrich_pinv=pinv))
    assert k1 < k0, f"enrichment did not reduce kappa: {k0:.2e} -> {k1:.2e}"


def test_random_useless_mode_does_not_help_much():
    """A random enrichment column does not collapse kappa like the worst mode does.

    Confirms the gain is from targeting the bad direction, not from merely adding a
    column (so the LEARNED amplitude has something real to do).
    """
    S, PI = _highcontrast_chain(contrast=1e3)
    M0 = make_two_level_minv(S, PI)
    vstar, k0 = _worst_mode(S, M0)
    rng = np.random.default_rng(7)
    Wr = rng.standard_normal((S.shape[0], 1))
    pinv = np.linalg.pinv(Wr.T @ (S @ Wr))
    k_rand = two_level_kappa_dense(S, make_two_level_minv(S, PI, enrich_W=Wr, enrich_pinv=pinv))
    # worst-mode enrichment must beat a random column on the SAME block
    Ws = vstar.reshape(-1, 1)
    pin = np.linalg.pinv(Ws.T @ (S @ Ws))
    k_star = two_level_kappa_dense(S, make_two_level_minv(S, PI, enrich_W=Ws, enrich_pinv=pin))
    assert k_star < k_rand, f"worst-mode ({k_star:.2e}) not better than random ({k_rand:.2e})"


def test_worst_mode_amplitude_label_shape_and_range():
    """The spectral worst-mode amplitude label is per coarse-node, normalized to [0,1]."""
    from fracturex.ml.spectral_labels import worst_mode_amplitude
    S, PI = _highcontrast_chain(contrast=1e3)
    amp = worst_mode_amplitude(S, PI, iters=40)
    assert amp.shape == (PI.shape[1],)
    assert amp.min() >= 0.0 and amp.max() == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(amp).all()


def test_top_k_worst_modes_orthonormal_and_convergent_M0():
    """top_k worst modes are orthonormal fine-space columns built on a CONVERGENT M0.

    The mode extraction iterates the error-propagation operator E = I - M0^{-1} S_b; M0
    must be convergent (rho(E) < 1) or the "worst modes" are garbage (the undamped
    omega=1 additive two-level DIVERGES on localized blocks, |lambda(E)|~2.3 -- D13_IMPL
    §6.7; omega=0.5 restores convergence). We check orthonormality and that the dominant
    Ritz value of E is < 1 (convergent). kappa-reduction is validated on the REAL
    localized operator (k-scan, §6.6/§6.8), not this benign synthetic chain.
    """
    from fracturex.ml.spectral_labels import top_k_worst_modes, make_two_level_minv
    S, PI = _highcontrast_chain(contrast=1e4)
    n = S.shape[0]
    k = min(8, n - 1)
    V = top_k_worst_modes(S, PI, k, iters=80, smoother_omega=0.5)
    assert V.shape == (n, k)
    assert np.allclose(V.T @ V, np.eye(k), atol=1e-8)  # orthonormal
    # E must be convergent at omega=0.5: dominant Ritz |lambda(E)| < 1.
    M0 = make_two_level_minv(S, PI, smoother_omega=0.5)
    EV = np.column_stack([V[:, j] - M0.matvec(S @ V[:, j]) for j in range(k)])
    ritz = np.abs(np.linalg.eigvals(V.T @ EV))
    assert ritz.max() < 1.0 + 1e-6, f"M0 not convergent: max|lambda(E)|={ritz.max():.3f}"


def test_enrichment_accepts_fine_modes_directly():
    """EnrichmentOperator uses fine-space (sgdof) modes directly, coarse (NN) prolonged."""
    import scipy.sparse as sp
    from fracturex.ml.coarse_space_enrich import EnrichmentOperator
    sgdof, nc, k = 12, 5, 2
    PI = sp.random(sgdof, nc, density=0.5, random_state=0).tocsr() + sp.eye(sgdof, nc).tocsr()
    rng = np.random.default_rng(0)
    Sb = np.eye(sgdof) * 2.0
    fine = rng.standard_normal((sgdof, k))   # fine-space modes
    enr = EnrichmentOperator.from_modes(fine, PI, lambda v, c: Sb @ v, gdim=1, sgdof=sgdof)
    assert enr is not None and enr.W.shape == (sgdof, k)
    assert np.allclose(enr.W, fine)  # used directly, NOT prolonged


# ---------------------------------------------------------------------------
# amplitude model
# ---------------------------------------------------------------------------

def test_amplitude_model_bounded_and_inference():
    from fracturex.ml.coarse_weight_model import (
        AmplitudeModelConfig, build_model, predict_amplitude,
    )
    cfg = AmplitudeModelConfig(a_max=0.7, feature_mean=[0.0] * N_FEATURES,
                               feature_std=[1.0] * N_FEATURES)
    model = build_model(cfg)
    phi, _ = _feat_block(30)
    a = predict_amplitude(model, phi)
    assert a.shape == (30,)
    assert a.min() >= 0.0 and a.max() <= 0.7 + 1e-9
    assert np.isfinite(a).all()


def test_amplitude_model_save_load(tmp_path):
    from fracturex.ml.coarse_weight_model import (
        AmplitudeModelConfig, build_model, predict_amplitude, save_model, load_model,
    )
    cfg = AmplitudeModelConfig(a_max=1.0)
    model = build_model(cfg)
    phi, _ = _feat_block(20)
    a0 = predict_amplitude(model, phi)
    p = tmp_path / "amp.pt"
    save_model(model, cfg, p)
    model2, _ = load_model(p)
    a1 = predict_amplitude(model2, phi)
    assert np.allclose(a0, a1, atol=1e-12)


# ---------------------------------------------------------------------------
# training smoke + isolation
# ---------------------------------------------------------------------------

def test_training_smoke_loss_decreases():
    from fealpy.backend import backend_manager as bm
    from fracturex.ml.datasets import FeatureSample, split_by_predicate
    from fracturex.ml.train_coarse_space import train_amplitude_model, TrainConfig

    rng = np.random.default_rng(0)
    samples = []
    for s in range(4):
        phi, _ = _feat_block(60)
        phi = phi + 0.01 * rng.standard_normal(phi.shape)
        samples.append(FeatureSample(
            phi=bm.asarray(phi), node=bm.asarray(rng.standard_normal((60, 2))),
            case="model0", hmin=0.025, l0=0.02, maxd=0.9 + 0.01 * s, eps_g=1e-6,
            source=f"syn{s}"))
    split = split_by_predicate(samples, lambda smp: smp.maxd > 0.92)
    _, hist = train_amplitude_model(split, TrainConfig(epochs=80, lr=5e-3))
    assert hist["train_loss"][-1] < hist["train_loss"][0], "train loss did not decrease"
    assert hist["train_loss"][-1] < 0.1


def test_training_uses_stored_target_over_heuristic():
    """When samples carry a stored spectral target, training regresses to IT (not the
    feature heuristic). We give a target uncorrelated with the heuristic and check the
    trained model predicts the target, not ideal_interface_amplitude.
    """
    from fealpy.backend import backend_manager as bm
    from fracturex.ml.datasets import FeatureSample, split_by_predicate
    from fracturex.ml.train_coarse_space import train_amplitude_model, TrainConfig, _stack_xy
    from fracturex.ml.spectral_labels import ideal_interface_amplitude

    rng = np.random.default_rng(1)
    samples = []
    for s in range(3):
        phi, _ = _feat_block(80)
        phi = phi + 0.01 * rng.standard_normal(phi.shape)
        # target = a fixed smooth pattern, deliberately != heuristic
        tgt = np.clip(0.5 + 0.4 * np.sin(np.linspace(0, 6, 80)), 0, 1)
        samples.append(FeatureSample(
            phi=bm.asarray(phi), node=bm.asarray(rng.standard_normal((80, 2))),
            case="model0", hmin=0.025, l0=0.02, maxd=0.95, eps_g=1e-6,
            source=f"syn{s}", target=bm.asarray(tgt)))
    # _stack_xy must pick the stored target, not the heuristic
    _, Y = _stack_xy(samples, use_target=True)
    heur = ideal_interface_amplitude(np.asarray(bm.to_numpy(samples[0].phi)))
    assert not np.allclose(Y[:80], heur), "training used heuristic instead of stored target"
    split = split_by_predicate(samples, lambda smp: False)
    model, hist = train_amplitude_model(split, TrainConfig(epochs=150, lr=5e-3))
    assert hist["train_loss"][-1] < 0.05


def test_l2beta_isolation_no_solver():
    import subprocess
    import sys
    code = (
        "import sys; import fracturex.ml.spectral_labels, fracturex.ml.coarse_weight_model,"
        "fracturex.ml.train_coarse_space;"
        "assert 'fracturex.utilfuc.linear_solvers' not in sys.modules; print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}{r.stderr}"
