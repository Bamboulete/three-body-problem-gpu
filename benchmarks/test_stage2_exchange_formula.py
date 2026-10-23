"""
STUFE-2 TRINITÄTS-THEORIE: DIE PARTNER-TAUSCH-FORMEL
Erweiterung der Master-Formel um den analytischen Hills-Exchange-Mechanismus.

Ziel:
1. Kann die Formel bei t=0 vorhersagen, OB ein Partner-Tausch stattfindet?
2. Wenn ja: Sagt sie korrekt voraus, dass Stern B statt Stern C herausfliegt?
3. Wie stark verbessert sich die Trefferquote und der Zyklenfehler?
"""

import time
import sys
import cupy as cp
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mega_benchmark_10m import cuda_kernel_source

def test_stage2_exchange():
    N = 500000
    print(f"=== TEST DER STUFE-2 PARTNER-TAUSCH-THEORIE ({N:,} SYSTEME) ===")
    
    module = cp.RawModule(code=cuda_kernel_source, options=('--use_fast_math',))
    kernel = module.get_function('simulate_3body_batch')
    
    cp.random.seed(777)
    m = cp.empty((N, 3), dtype=cp.float32)
    m[:, 0] = cp.random.uniform(1.0, 3.5, N, dtype=cp.float32)
    m[:, 1] = cp.random.uniform(0.7, 2.5, N, dtype=cp.float32)
    m[:, 2] = cp.random.uniform(0.3, 1.6, N, dtype=cp.float32)

    scale = cp.random.uniform(1.2, 3.0, (N, 1), dtype=cp.float32)
    r = cp.empty((N, 3, 3), dtype=cp.float32)
    r[:, :, 0] = cp.random.uniform(-2.0, 2.0, (N, 3), dtype=cp.float32) * scale
    r[:, :, 1] = cp.random.uniform(-2.0, 2.0, (N, 3), dtype=cp.float32) * scale
    r[:, :, 2] = cp.random.uniform(-0.5, 0.5, (N, 3), dtype=cp.float32) * scale

    v_scale = cp.random.uniform(0.35, 0.85, (N, 1), dtype=cp.float32)
    v = cp.empty((N, 3, 3), dtype=cp.float32)
    v[:, :, 0] = cp.random.uniform(-1.0, 1.0, (N, 3), dtype=cp.float32) * v_scale
    v[:, :, 1] = cp.random.uniform(-1.0, 1.0, (N, 3), dtype=cp.float32) * v_scale
    v[:, :, 2] = cp.random.uniform(-0.3, 0.3, (N, 3), dtype=cp.float32) * v_scale

    M_tot = m.sum(axis=1, keepdims=True)
    p_cm = (m[:, :, None] * v).sum(axis=1) / M_tot
    v -= p_cm[:, None, :]

    # 1. Bestimme das initiale Paar (Doppelstern bei t=0) und den Eindringling C
    eps2 = 0.05**2
    d01_2 = ((r[:, 1] - r[:, 0])**2).sum(axis=1) + eps2
    d12_2 = ((r[:, 2] - r[:, 1])**2).sum(axis=1) + eps2
    d20_2 = ((r[:, 0] - r[:, 2])**2).sum(axis=1) + eps2

    # Finde kleinsten Abstand bei t=0
    cond01 = (d01_2 <= d12_2) & (d01_2 <= d20_2)
    cond12 = (d12_2 <= d01_2) & (d12_2 <= d20_2) & ~cond01
    cond20 = ~cond01 & ~cond12

    # Eindringling C ist der dritte Stern
    intruder_c = cp.where(cond01, 2, cp.where(cond12, 0, 1)).astype(cp.int32)
    
    # Doppelstern-Partner (b0, b1)
    b0 = cp.where(cond01, 0, cp.where(cond12, 1, 2)).astype(cp.int32)
    b1 = cp.where(cond01, 1, cp.where(cond12, 2, 0)).astype(cp.int32)

    m_c = cp.take_along_axis(m, intruder_c[:, None], axis=1)[:, 0]
    m_b0 = cp.take_along_axis(m, b0[:, None], axis=1)[:, 0]
    m_b1 = cp.take_along_axis(m, b1[:, None], axis=1)[:, 0]

    # Welcher der beiden Doppelstern-Partner ist der leichtere (B_leicht)?
    m_heavy = cp.maximum(m_b0, m_b1)
    m_light = cp.minimum(m_b0, m_b1)
    b_light_idx = cp.where(m_b0 <= m_b1, b0, b1)
    b_heavy_idx = cp.where(m_b0 > m_b1, b0, b1)

    # 2. DAS HILLS-EXCHANGE-KRITERIUM BEI t = 0:
    # Wenn Eindringling C schwerer ist als der leichtere Partner (m_C > m_light):
    # Dann verdrängt C den leichten Partner B_leicht!
    # Prognose: Stern B_leicht fliegt raus!
    # Andernfalls: Kein Tausch, Stern C fliegt raus!
    predict_exchange = (m_c > m_light)
    predicted_ejectee_stage2 = cp.where(predict_exchange, b_light_idx, intruder_c)

    # 3. Klassischer Ansatz (rein starrer Kandidat bei t=0)
    # Frühere Formel: Basierend rein auf kinetischer Energie / Potenzial
    phi0 = 1.0 * (m[:, 1]/cp.sqrt(d01_2) + m[:, 2]/cp.sqrt(d20_2))
    phi1 = 1.0 * (m[:, 0]/cp.sqrt(d01_2) + m[:, 2]/cp.sqrt(d12_2))
    phi2 = 1.0 * (m[:, 0]/cp.sqrt(d20_2) + m[:, 1]/cp.sqrt(d12_2))
    phi_all = cp.stack([phi0, phi1, phi2], axis=1)
    v_sq = (v**2).sum(axis=2)
    xi = v_sq / (2.0 * cp.maximum(0.01, phi_all))
    score = ((M_tot / m)**1.5) * (1.0 + 1.2 * xi)
    cand_stage1 = cp.argmax(score, axis=1).astype(cp.int32)

    # 4. Simulation auf der GPU ausführen
    t_out = cp.zeros(N, dtype=cp.float32)
    cycles_out = cp.zeros(N, dtype=cp.int32)
    ej_out = cp.zeros(N, dtype=cp.int32)

    grid = (N + 255) // 256
    cp.cuda.Stream.null.synchronize()
    t_start = time.perf_counter()
    kernel((grid,), (256,), (m, r, v, t_out, cycles_out, ej_out, N, cp.float32(0.004), np.float32(90.0)))
    cp.cuda.Stream.null.synchronize()
    print(f"CUDA-Lauf in {time.perf_counter() - t_start:.2f} s abgeschlossen.\n")

    # 5. Auswertung der Trefferquoten
    valid = (t_out >= 4.0) & (t_out < 90.0) & (ej_out >= 0)
    num_valid = int(valid.sum())
    
    ej_real = ej_out[valid].get()
    c_stage1 = cand_stage1[valid].get()
    c_stage2 = predicted_ejectee_stage2[valid].get()
    cyc = cycles_out[valid].get()
    is_ex = predict_exchange[valid].get()

    hit_stage1 = np.mean(ej_real == c_stage1) * 100.0
    hit_stage2 = np.mean(ej_real == c_stage2) * 100.0

    print("=" * 86)
    print("  VERGLEICH: KLASSISCHES t=0 MODELL vs. STUFE-2 PARTNER-TAUSCH-FORMEL")
    print("=" * 86)
    print(f"Auswertbare Systeme            : {num_valid:,} von {N:,}")
    print(f"Systeme mit prognostiziertem Tausch: {np.sum(is_ex):,} ({np.mean(is_ex)*100:.1f}%)")
    print("-" * 86)
    print(f"Trefferquote Stufe 1 (Klassisch t=0) : {hit_stage1:.2f}%")
    print(f"Trefferquote Stufe 2 (Exchange-Modell) : {hit_stage2:.2f}%")
    print("=" * 86)

    # Aufschlüsselung nach Zyklen N
    print("\nTrefferquote aufgeschlüsselt nach Lebensdauer (Zyklen N):")
    print(" Zyklus N │ Treffer Stufe 1 (Klassisch) │ Treffer Stufe 2 (Exchange) │ Gewinn / Vorteil")
    print("──────────┼─────────────────────────────┼────────────────────────────┼───────────────────")
    for n in range(1, 11):
        mask_n = (cyc == n)
        if np.sum(mask_n) > 0:
            h1 = np.mean(ej_real[mask_n] == c_stage1[mask_n]) * 100.0
            h2 = np.mean(ej_real[mask_n] == c_stage2[mask_n]) * 100.0
            diff = h2 - h1
            sign = "+" if diff >= 0 else ""
            print(f"   N = {n:2d}   │           {h1:5.1f}%           │          {h2:5.1f}%         │     {sign}{diff:4.1f}%")

    mask_deep = (cyc > 10)
    h1_deep = np.mean(ej_real[mask_deep] == c_stage1[mask_deep]) * 100.0
    h2_deep = np.mean(ej_real[mask_deep] == c_stage2[mask_deep]) * 100.0
    diff_d = h2_deep - h1_deep
    sign_d = "+" if diff_d >= 0 else ""
    print(f"   N > 10 │           {h1_deep:5.1f}%           │          {h2_deep:5.1f}%         │     {sign_d}{diff_d:4.1f}%")
    print("=" * 86)

if __name__ == "__main__":
    test_stage2_exchange()
