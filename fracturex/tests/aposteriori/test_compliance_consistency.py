# fracturex/tests/aposteriori/test_compliance_consistency.py
"""T0: 柔度 / 退化权一致性（THEORY (2)，DESIGN 测试矩阵 T0）。

验证 equilibrated_estimator 的代数核正确，不依赖 FE 空间：
  (a) A(d) C_d = Id     退化柔度与退化弹性互逆（Voigt round-trip 机器精度）
  (b) C^{-1} C = Id      未退化层 round-trip
  (c) 估计子在 σ_h = C_d ε(u_h) 时 η ≡ 0（残差为零 ⇒ 指示子为零）
  (d) degradation 正性 + k_res<=0 抛错
判据：所有 round-trip rel-error < 1e-12；η < 1e-12。

运行：
  source miniconda3/etc/profile.d/conda.sh && conda activate py312
  PYTHONPATH=tian/fealpy:tian/fracturex python tian/fracturex/tests/aposteriori/test_compliance_consistency.py
"""
from __future__ import annotations

from fealpy.backend import backend_manager as bm

from fracturex.adaptivity.equilibrated_estimator import (
    stress_voigt_from_strain,
    compliance_apply_voigt,
    voigt_inner,
    degradation,
    equilibrated_indicator,
)

LAM, MU = 121.15, 80.77  # 典型钢材 Lamé（平面应变）


def _rel(a, b):
    """相对误差 ‖a-b‖ / max(‖b‖, 1)。"""
    num = float(bm.sqrt(bm.sum((a - b) ** 2)))
    den = max(float(bm.sqrt(bm.sum(b ** 2))), 1.0)
    return num / den


def test_compliance_roundtrip():
    """(b) C^{-1} (C ε) = ε，对随机应变场。"""
    bm.set_backend("numpy")
    eps = bm.array([[0.3, -0.2, 0.5], [1.0, 0.0, -0.7], [-0.4, 0.9, 0.1]])
    sig = stress_voigt_from_strain(eps, LAM, MU)
    eps_back = compliance_apply_voigt(sig, LAM, MU)
    rel = _rel(eps_back, eps)
    assert rel < 1e-12, f"C^-1 C != Id, rel={rel:.2e}"
    print(f"[T0.b] C^-1 C = Id  rel={rel:.2e}  OK")


def test_degraded_roundtrip():
    """(a) A(d) C_d = Id：A(d)=g^{-1}C^{-1}, C_d=g C ⇒ g^{-1}C^{-1}(g C ε)=ε。"""
    bm.set_backend("numpy")
    for d in (0.0, 0.3, 0.7, 0.99, 1.0):
        for k_res in (1e-3, 1e-7):
            g = degradation(bm.array(d), k_res)
            eps = bm.array([[0.3, -0.2, 0.5], [1.0, 0.0, -0.7]])
            Cd_eps = float(g) * stress_voigt_from_strain(eps, LAM, MU)      # C_d ε
            eps_back = compliance_apply_voigt(Cd_eps, LAM, MU) / float(g)   # A(d) (C_d ε)
            rel = _rel(eps_back, eps)
            assert rel < 1e-12, f"A(d)C_d != Id at d={d},k={k_res}, rel={rel:.2e}"
    print("[T0.a] A(d) C_d = Id  (扫 d∈{0,.3,.7,.99,1}, k∈{1e-3,1e-7})  OK")


def test_zero_residual_zero_eta():
    """(c) σ_h = C_d ε(u_h) ⇒ 残差为零 ⇒ η = 0。"""
    bm.set_backend("numpy")
    NC, NQ = 5, 4
    # 随机位移梯度
    rng_grad = bm.array(
        [[[[0.1 * (c + q), -0.05 * q], [0.02 * c, 0.08 * (c - q)]]
          for q in range(NQ)] for c in range(NC)]
    )  # (NC,NQ,2,2)
    d_qp = bm.array([[0.1 * c + 0.05 * q for q in range(NQ)] for c in range(NC)])
    k_res = 1e-5
    g = degradation(d_qp, k_res)                                    # (NC,NQ)
    # 由 ε(u_h) 构造正好平衡的 σ_h = C_d ε(u_h)
    from fracturex.adaptivity.equilibrated_estimator import strain_to_voigt
    eps_v = strain_to_voigt(rng_grad)
    sigmah_qp = g[..., None] * stress_voigt_from_strain(eps_v, LAM, MU)
    weights = bm.array([0.25] * NQ)
    cellmeasure = bm.array([0.5] * NC)
    out = equilibrated_indicator(None, rng_grad, sigmah_qp, d_qp,
                                 lam=LAM, mu=MU, k_res=k_res,
                                 weights=weights, cellmeasure=cellmeasure)
    assert out["eta"] < 1e-12, f"zero-residual eta != 0: {out['eta']:.2e}"
    print(f"[T0.c] σ_h=C_d ε(u_h) ⇒ η={out['eta']:.2e}  OK")


def test_degradation_positivity():
    """(d) g>0 恒成立；k_res<=0 抛 ValueError。"""
    bm.set_backend("numpy")
    d = bm.array([0.0, 0.5, 1.0])
    g = degradation(d, 1e-6)
    assert float(bm.min(g)) > 0.0
    try:
        degradation(d, 0.0)
    except ValueError:
        print("[T0.d] g>0 且 k_res<=0 抛错  OK")
        return
    raise AssertionError("k_res=0 should raise ValueError")


if __name__ == "__main__":
    test_compliance_roundtrip()
    test_degraded_roundtrip()
    test_zero_residual_zero_eta()
    test_degradation_positivity()
    print("\n[T0] ALL PASS")
