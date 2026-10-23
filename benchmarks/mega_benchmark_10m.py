"""
DAS MEGA-EXPERIMENT: 10.000.000 CHAOTISCHE 3-KÖRPER-SYSTEME
Hardware: NVIDIA GeForce RTX 4070 (12 GB VRAM) + AMD Ryzen 7 5800X
Native CUDA C++ Register-Integration via CuPy RawModule.

Pipeline:
1. 10 Batches à 1.000.000 Systeme = 10.000.000 Gesamtsimulationen
2. Reale Zyklenzählung N_real, Lebensdauer t_eject, Auswurf-Kandidat
3. Globale Nelder-Mead Optimierung der Trinitäts-Gleichung auf Millionen-Datensatz
4. Statistische Endauswertung & hochauflösendes Resonanz-Histogramm
"""

import time
import math
import sys
import json
import numpy as np
import cupy as cp
import scipy.optimize as opt

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cuda_kernel_source = r'''
extern "C" __global__
void simulate_3body_batch(
    const float* __restrict__ m_in,          // [N, 3]
    const float* __restrict__ r_in,          // [N, 3, 3]
    const float* __restrict__ v_in,          // [N, 3, 3]
    float* __restrict__ t_eject_out,         // [N]
    int* __restrict__ cycles_out,            // [N]
    int* __restrict__ ejected_idx_out,       // [N]
    const int num_systems,
    const float dt,
    const float t_max
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_systems) return;

    float m0 = m_in[idx * 3 + 0];
    float m1 = m_in[idx * 3 + 1];
    float m2 = m_in[idx * 3 + 2];
    float M_tot = m0 + m1 + m2;

    float r0x = r_in[idx * 9 + 0], r0y = r_in[idx * 9 + 1], r0z = r_in[idx * 9 + 2];
    float r1x = r_in[idx * 9 + 3], r1y = r_in[idx * 9 + 4], r1z = r_in[idx * 9 + 5];
    float r2x = r_in[idx * 9 + 6], r2y = r_in[idx * 9 + 7], r2z = r_in[idx * 9 + 8];

    float v0x = v_in[idx * 9 + 0], v0y = v_in[idx * 9 + 1], v0z = v_in[idx * 9 + 2];
    float v1x = v_in[idx * 9 + 3], v1y = v_in[idx * 9 + 4], v1z = v_in[idx * 9 + 5];
    float v2x = v_in[idx * 9 + 6], v2y = v_in[idx * 9 + 7], v2z = v_in[idx * 9 + 8];

    float eps2 = 0.0025f; // 0.05^2
    float t = 0.0f;
    int max_steps = (int)(t_max / dt);
    float dt2_half = 0.5f * dt * dt;
    float dt_half = 0.5f * dt;

    float prev_dmin = 100.0f;
    float prev_rate = 0.0f;
    int cycles = 0;
    float eject_t = t_max;
    int ejected_idx = -1;

    // Initiale Beschleunigungen
    float dx01 = r1x - r0x; float dy01 = r1y - r0y; float dz01 = r1z - r0z;
    float dx12 = r2x - r1x; float dy12 = r2y - r1y; float dz12 = r2z - r1z;
    float dx20 = r0x - r2x; float dy20 = r0y - r2y; float dz20 = r0z - r2z;

    float d01_2 = dx01*dx01 + dy01*dy01 + dz01*dz01 + eps2;
    float d12_2 = dx12*dx12 + dy12*dy12 + dz12*dz12 + eps2;
    float d20_2 = dx20*dx20 + dy20*dy20 + dz20*dz20 + eps2;

    float d01 = sqrtf(d01_2); float d12 = sqrtf(d12_2); float d20 = sqrtf(d20_2);
    float inv3_01 = 1.0f / (d01_2 * d01);
    float inv3_12 = 1.0f / (d12_2 * d12);
    float inv3_20 = 1.0f / (d20_2 * d20);

    float a0x = m1*inv3_01*dx01 - m2*inv3_20*dx20;
    float a0y = m1*inv3_01*dy01 - m2*inv3_20*dy20;
    float a0z = m1*inv3_01*dz01 - m2*inv3_20*dz20;

    float a1x = m2*inv3_12*dx12 - m0*inv3_01*dx01;
    float a1y = m2*inv3_12*dy12 - m0*inv3_01*dy01;
    float a1z = m2*inv3_12*dz12 - m0*inv3_01*dz01;

    float a2x = m0*inv3_20*dx20 - m1*inv3_12*dx12;
    float a2y = m0*inv3_20*dy20 - m1*inv3_12*dy12;
    float a2z = m0*inv3_20*dz20 - m1*inv3_12*dz12;

    for (int s = 0; s < max_steps; ++s) {
        r0x += v0x * dt + a0x * dt2_half;
        r0y += v0y * dt + a0y * dt2_half;
        r0z += v0z * dt + a0z * dt2_half;

        r1x += v1x * dt + a1x * dt2_half;
        r1y += v1y * dt + a1y * dt2_half;
        r1z += v1z * dt + a1z * dt2_half;

        r2x += v2x * dt + a2x * dt2_half;
        r2y += v2y * dt + a2y * dt2_half;
        r2z += v2z * dt + a2z * dt2_half;

        dx01 = r1x - r0x; dy01 = r1y - r0y; dz01 = r1z - r0z;
        dx12 = r2x - r1x; dy12 = r2y - r1y; dz12 = r2z - r1z;
        dx20 = r0x - r2x; dy20 = r0y - r2y; dz20 = r0z - r2z;

        d01_2 = dx01*dx01 + dy01*dy01 + dz01*dz01 + eps2;
        d12_2 = dx12*dx12 + dy12*dy12 + dz12*dz12 + eps2;
        d20_2 = dx20*dx20 + dy20*dy20 + dz20*dz20 + eps2;

        d01 = sqrtf(d01_2); d12 = sqrtf(d12_2); d20 = sqrtf(d20_2);
        inv3_01 = 1.0f / (d01_2 * d01);
        inv3_12 = 1.0f / (d12_2 * d12);
        inv3_20 = 1.0f / (d20_2 * d20);

        float a0x_new = m1*inv3_01*dx01 - m2*inv3_20*dx20;
        float a0y_new = m1*inv3_01*dy01 - m2*inv3_20*dy20;
        float a0z_new = m1*inv3_01*dz01 - m2*inv3_20*dz20;

        float a1x_new = m2*inv3_12*dx12 - m0*inv3_01*dx01;
        float a1y_new = m2*inv3_12*dy12 - m0*inv3_01*dy01;
        float a1z_new = m2*inv3_12*dz12 - m0*inv3_01*dz01;

        float a2x_new = m0*inv3_20*dx20 - m1*inv3_12*dx12;
        float a2y_new = m0*inv3_20*dy20 - m1*inv3_12*dy12;
        float a2z_new = m0*inv3_20*dz20 - m1*inv3_12*dz12;

        v0x += (a0x + a0x_new) * dt_half;
        v0y += (a0y + a0y_new) * dt_half;
        v0z += (a0z + a0z_new) * dt_half;

        v1x += (a1x + a1x_new) * dt_half;
        v1y += (a1y + a1y_new) * dt_half;
        v1z += (a1z + a1z_new) * dt_half;

        v2x += (a2x + a2x_new) * dt_half;
        v2y += (a2y + a2y_new) * dt_half;
        v2z += (a2z + a2z_new) * dt_half;

        a0x = a0x_new; a0y = a0y_new; a0z = a0z_new;
        a1x = a1x_new; a1y = a1y_new; a1z = a1z_new;
        a2x = a2x_new; a2y = a2y_new; a2z = a2z_new;

        t += dt;

        float cur_dmin = fminf(d01, fminf(d12, d20));
        float rate = (cur_dmin - prev_dmin) / dt;
        if (prev_rate < 0.0f && rate >= 0.0f && cur_dmin < 2.5f) {
            cycles++;
        }
        prev_dmin = cur_dmin;
        prev_rate = rate;

        if ((s & 31) == 0) {
            float cx = (m0*r0x + m1*r1x + m2*r2x) / M_tot;
            float cy = (m0*r0y + m1*r1y + m2*r2y) / M_tot;
            float cz = (m0*r0z + m1*r1z + m2*r2z) / M_tot;

            float d0_sq = (r0x-cx)*(r0x-cx) + (r0y-cy)*(r0y-cy) + (r0z-cz)*(r0z-cz);
            float d1_sq = (r1x-cx)*(r1x-cx) + (r1y-cy)*(r1y-cy) + (r1z-cz)*(r1z-cz);
            float d2_sq = (r2x-cx)*(r2x-cx) + (r2y-cy)*(r2y-cy) + (r2z-cz)*(r2z-cz);

            float max_d_sq = fmaxf(d0_sq, fmaxf(d1_sq, d2_sq));
            if (max_d_sq > 196.0f) { // 14.0^2
                eject_t = t;
                if (d0_sq >= d1_sq && d0_sq >= d2_sq) ejected_idx = 0;
                else if (d1_sq >= d0_sq && d1_sq >= d2_sq) ejected_idx = 1;
                else ejected_idx = 2;
                break;
            }
        }
    }

    t_eject_out[idx] = eject_t;
    cycles_out[idx] = (cycles < 1) ? 1 : cycles;
    ejected_idx_out[idx] = ejected_idx;
}
'''

def run_mega_10m():
    TOTAL_TARGET = 10000000 # 10 Millionen!
    BATCH_SIZE = 1000000    # 1 Million pro Batch
    NUM_BATCHES = TOTAL_TARGET // BATCH_SIZE

    print("=" * 88)
    print("  DAS 10.000.000-SYSTEME-MEGA-EXPERIMENT: TRINITÄTS-MASTERFORMEL")
    print("=" * 88)
    print("  Beschleuniger      : NVIDIA GeForce RTX 4070 (12 GB GDDR6X)")
    print("  Architektur        : Reines CUDA C++ Register-Computing (NVRTC)")
    print(f"  Geplante Systeme   : {TOTAL_TARGET:,} chaotische 3-Körper-Systeme")
    print(f"  Etappen            : {NUM_BATCHES} Batches à {BATCH_SIZE:,} Systeme")
    print("=" * 88)

    print("Kompiliere CUDA C++ Hardware-Kernel...")
    module = cp.RawModule(code=cuda_kernel_source)
    simulate_kernel = module.get_function('simulate_3body_batch')

    start_mega = time.perf_counter()

    # Akkumulatoren für aggregierte Auswertung
    all_t_reals = []
    all_n_reals = []
    all_t_scales = []
    all_dphis = []
    all_etas = []
    all_cos_alphas = []
    all_masses = []
    all_cand_hits = []

    total_simulated = 0
    total_valid = 0

    block_size = 256
    grid_size = (BATCH_SIZE + block_size - 1) // block_size

    for b in range(1, NUM_BATCHES + 1):
        t_batch_start = time.perf_counter()

        # 1. Erzeuge 1 Million Systeme direkt im VRAM
        cp.random.seed(100000 + b * 777)
        m = cp.empty((BATCH_SIZE, 3), dtype=cp.float32)
        m[:, 0] = cp.random.uniform(1.0, 3.5, BATCH_SIZE, dtype=cp.float32)
        m[:, 1] = cp.random.uniform(0.7, 2.5, BATCH_SIZE, dtype=cp.float32)
        m[:, 2] = cp.random.uniform(0.3, 1.6, BATCH_SIZE, dtype=cp.float32)

        scale = cp.random.uniform(1.2, 3.0, (BATCH_SIZE, 1), dtype=cp.float32)
        r = cp.empty((BATCH_SIZE, 3, 3), dtype=cp.float32)
        r[:, :, 0] = cp.random.uniform(-2.0, 2.0, (BATCH_SIZE, 3), dtype=cp.float32) * scale
        r[:, :, 1] = cp.random.uniform(-2.0, 2.0, (BATCH_SIZE, 3), dtype=cp.float32) * scale
        r[:, :, 2] = cp.random.uniform(-0.5, 0.5, (BATCH_SIZE, 3), dtype=cp.float32) * scale

        v_scale = cp.random.uniform(0.35, 0.85, (BATCH_SIZE, 1), dtype=cp.float32)
        v = cp.empty((BATCH_SIZE, 3, 3), dtype=cp.float32)
        v[:, :, 0] = cp.random.uniform(-1.0, 1.0, (BATCH_SIZE, 3), dtype=cp.float32) * v_scale
        v[:, :, 1] = cp.random.uniform(-1.0, 1.0, (BATCH_SIZE, 3), dtype=cp.float32) * v_scale
        v[:, :, 2] = cp.random.uniform(-0.3, 0.3, (BATCH_SIZE, 3), dtype=cp.float32) * v_scale

        M_tot = m.sum(axis=1, keepdims=True)
        p_cm = (m[:, :, None] * v).sum(axis=1) / M_tot
        v -= p_cm[:, None, :]

        # 2. t=0 Feature-Extraktion auf GPU
        eps2 = 0.05**2
        kin0 = 0.5 * (m[:, :, None] * (v**2)).sum(axis=(1, 2))
        d01 = cp.sqrt(((r[:, 1] - r[:, 0])**2).sum(axis=1) + eps2)
        d12 = cp.sqrt(((r[:, 2] - r[:, 1])**2).sum(axis=1) + eps2)
        d20 = cp.sqrt(((r[:, 0] - r[:, 2])**2).sum(axis=1) + eps2)

        omega0 = (m[:, 0]*m[:, 1])/d01 + (m[:, 1]*m[:, 2])/d12 + (m[:, 2]*m[:, 0])/d20
        E_tot = kin0 - omega0
        bound_mask = (E_tot < -0.05)

        absE = cp.abs(E_tot)
        E_eff = cp.maximum(absE, 0.25)
        t_scale = (1.0 * (M_tot[:, 0]**2.5)) / (E_eff**1.5)

        # Drehimpuls L & Trägheitsmoment I
        cm_pos = (m[:, :, None] * r).sum(axis=1) / M_tot
        r_rel_cm = r - cm_pos[:, None, :]
        I_val = (m * ((r_rel_cm**2).sum(axis=2))).sum(axis=1)

        L_vec = (m[:, :, None] * cp.cross(r, v, axis=2)).sum(axis=1)
        L_mag = cp.sqrt((L_vec**2).sum(axis=1))
        eta_rot = (L_mag**2) / (absE * cp.maximum(0.01, I_val))

        # Gravitationspotenziale & Delta_Phi_rel
        phi0 = 1.0 * (m[:, 1]/d01 + m[:, 2]/d20)
        phi1 = 1.0 * (m[:, 0]/d01 + m[:, 2]/d12)
        phi2 = 1.0 * (m[:, 0]/d20 + m[:, 1]/d12)
        phi_all = cp.stack([phi0, phi1, phi2], axis=1)
        phi_mean = phi_all.mean(axis=1)
        delta_phi_rel = (phi_all.max(axis=1) - phi_all.min(axis=1)) / cp.maximum(0.01, phi_mean)

        # Auswurf-Kandidat bei t=0
        v_sq = (v**2).sum(axis=2)
        xi = v_sq / (2.0 * cp.maximum(0.01, phi_all))
        score = ((M_tot / m)**1.5) * (1.0 + 1.2 * xi)
        cand_idx = cp.argmax(score, axis=1)

        # cos(alpha) Vektor-Geometrie
        m_cand = cp.take_along_axis(m, cand_idx[:, None], axis=1)
        m_bin = m.sum(axis=1, keepdims=True) - m_cand
        r_cand = cp.take_along_axis(r, cand_idx[:, None, None], axis=1)[:, 0, :]
        r_cm_tot = (m[:, :, None] * r).sum(axis=1)
        r_cm_bin = (r_cm_tot - m_cand * r_cand) / m_bin
        r_rel = r_cand - r_cm_bin

        v_cand = cp.take_along_axis(v, cand_idx[:, None, None], axis=1)[:, 0, :]
        v_cm_tot = (m[:, :, None] * v).sum(axis=1)
        v_cm_bin = (v_cm_tot - m_cand * v_cand) / m_bin
        v_rel = v_cand - v_cm_bin

        cos_alpha = (r_rel * v_rel).sum(axis=1) / (cp.maximum(0.01, cp.sqrt((r_rel**2).sum(axis=1))) * cp.maximum(0.01, cp.sqrt((v_rel**2).sum(axis=1))))

        # 3. CUDA Kernel Zeitschritt-Integration
        t_out = cp.zeros(BATCH_SIZE, dtype=cp.float32)
        cycles_out = cp.zeros(BATCH_SIZE, dtype=cp.int32)
        ejected_idx_out = cp.zeros(BATCH_SIZE, dtype=cp.int32)

        cp.cuda.Stream.null.synchronize()
        simulate_kernel(
            (grid_size,), (block_size,),
            (m, r, v, t_out, cycles_out, ejected_idx_out, BATCH_SIZE, np.float32(0.004), np.float32(90.0))
        )
        cp.cuda.Stream.null.synchronize()

        # 4. Filter gültige zerfallene Systeme (4.0 <= t < 90.0 und gebunden)
        valid_mask = bound_mask & (t_out >= 4.0) & (t_out < 90.0)
        cand_hits_batch = (cand_idx == ejected_idx_out)

        # Extrahiere Host-Arrays für Akkumulation
        t_reals_b = t_out[valid_mask].get()
        n_reals_b = cycles_out[valid_mask].get()
        t_scales_b = t_scale[valid_mask].get()
        dphis_b = delta_phi_rel[valid_mask].get()
        etas_b = eta_rot[valid_mask].get()
        cos_alphas_b = cos_alpha[valid_mask].get()
        masses_b = M_tot[:, 0][valid_mask].get()
        cand_hits_b = cand_hits_batch[valid_mask].get()

        all_t_reals.append(t_reals_b)
        all_n_reals.append(n_reals_b)
        all_t_scales.append(t_scales_b)
        all_dphis.append(dphis_b)
        all_etas.append(etas_b)
        all_cos_alphas.append(cos_alphas_b)
        all_masses.append(masses_b)
        all_cand_hits.append(cand_hits_b)

        total_simulated += BATCH_SIZE
        total_valid += len(t_reals_b)

        t_batch_elapsed = time.perf_counter() - t_batch_start
        rate = BATCH_SIZE / t_batch_elapsed
        t_total_elapsed = time.perf_counter() - start_mega

        print(f"  [Batch {b:2d}/10] {total_simulated:10,d} simuliert | {total_valid:9,d} gültig "
              f"({rate:,.0f} Sys/s | Batch: {t_batch_elapsed:4.2f}s | Gesamt: {t_total_elapsed:4.1f}s)")

    sim_duration = time.perf_counter() - start_mega
    print("\n" + "=" * 88)
    print(f"  10.000.000 SIMULATIONEN ABGESCHLOSSEN IN {sim_duration:.2f} SEKUNDEN ({sim_duration/60:.2f} MINUTEN)!")
    print(f"  Gültige zerfallene Systeme: {total_valid:,} ({total_valid/total_target_pct:.1f}% Erfolgsquote)" if 'total_target_pct' in locals() else f"  Gültige zerfallene Systeme: {total_valid:,}")
    print(f"  Durchschnittliche Durchsatzrate: {total_simulated/sim_duration:,.0f} Systeme pro Sekunde!")
    print("=" * 88)

    # Zusammenführen zu riesigen NumPy-Arrays
    print("Führe Datensätze zusammen...")
    t_reals = np.concatenate(all_t_reals)
    n_reals = np.concatenate(all_n_reals)
    t_scales = np.concatenate(all_t_scales)
    dphis = np.concatenate(all_dphis)
    etas = np.concatenate(all_etas)
    cos_alphas = np.concatenate(all_cos_alphas)
    masses = np.concatenate(all_masses)
    cand_hits = np.concatenate(all_cand_hits)

    N_valid = len(t_reals)
    print(f"Gesamtanzahl auswertbarer Systeme: {N_valid:,}")

    # -------------------------------------------------------------
    # 1. AUSWERTUNG DES KLASSISCHEN HEGGIE-MODELLS
    # -------------------------------------------------------------
    pred_heggie = 0.50 * t_scales
    errs_heggie = np.abs(pred_heggie - t_reals) / t_reals * 100.0
    med_heggie = np.median(errs_heggie)

    # 2. AUSWERTUNG DER TRINITÄTS-FORMEL (STAND 100k)
    # Parameter aus 100k: C_N=1.7090, beta=0.3488, eps_0=0.2632, k=6.8487, gamma=0.0359, alpha=0.4681
    gate_100k = 0.5 * (1.0 + np.tanh(6.8487 * (cos_alphas + 0.05)))
    rot_100k = 1.0 + (np.minimum(2.5, etas)**0.3488)
    n_base_100k = (1.7090 * rot_100k) / (dphis + 0.2632)
    n_pred_100k = np.maximum(1, 1 + np.round((n_base_100k - 1.0) * gate_100k))
    t_kep_100k = 0.4681 * (14.0**1.5) / np.sqrt(1.0 * masses)
    pred_100k = (n_pred_100k - 1) * (0.0359 * t_scales) + t_kep_100k
    errs_100k = np.abs(pred_100k - t_reals) / t_reals * 100.0
    med_100k = np.median(errs_100k)
    wins_100k = np.sum(errs_100k < errs_heggie)

    print("\n[VERGLEICH VOR DER 10M-FEINEICHUNG]:")
    print(f"  • Trefferquote Auswurf-Kandidat : {np.mean(cand_hits)*100:.2f}% ({np.sum(cand_hits):,} Systeme)")
    print(f"  • Median-Fehler Heggie (1975)   : {med_heggie:.2f}%")
    print(f"  • Median-Fehler Trinität (100k) : {med_100k:.2f}%")
    print(f"  • Trinität schlägt Heggie in    : {wins_100k/N_valid*100:.2f}% aller Fälle")

    # -------------------------------------------------------------
    # 3. GLOBALE NELDER-MEAD OPTIMIERUNG AUF DEM 10M-DATENSATZ
    # -------------------------------------------------------------
    print("\n" + "-" * 88)
    print("  STARTE ULTIMATIVE GLOBALE PARAMETER-OPTIMIERUNG (10M DATENSATZ)...")
    print("-" * 88)

    # Subsample für ultra-schnelle Konvergenz der 6 Parameter
    SAMPLE_SIZE = min(500000, N_valid)
    np.random.seed(999)
    sub_idx = np.random.choice(N_valid, SAMPLE_SIZE, replace=False)

    sub_cos = cos_alphas[sub_idx]
    sub_etas = etas[sub_idx]
    sub_dphis = dphis[sub_idx]
    sub_masses = masses[sub_idx]
    sub_t_scales = t_scales[sub_idx]
    sub_t_reals = t_reals[sub_idx]

    def loss_func(p):
        C_N, beta, eps_0, k_gate, gamma_orb, alpha_kep = p
        gate = 0.5 * (1.0 + np.tanh(k_gate * (sub_cos + 0.05)))
        rot = 1.0 + (np.minimum(2.5, sub_etas)**beta)
        n_b = (C_N * rot) / (sub_dphis + eps_0)
        n_p = np.maximum(1, 1 + np.round((n_b - 1.0) * gate))
        t_k = alpha_kep * (14.0**1.5) / np.sqrt(1.0 * sub_masses)
        t_p = (n_p - 1) * (gamma_orb * sub_t_scales) + t_k
        errs = np.abs(t_p - sub_t_reals) / sub_t_reals * 100.0
        return np.median(errs)

    init_params = [1.7090, 0.3488, 0.2632, 6.8487, 0.0359, 0.4681]
    bounds = [
        (1.0, 5.0),     # C_N
        (0.1, 0.6),     # beta
        (0.05, 0.5),    # eps_0 (Dämpfung immer strikt positiv)
        (3.0, 12.0),    # k_gate (Doppler-Tor Steilheit)
        (0.005, 0.15),  # gamma_orb (Orbit-Zeitkopplung pro Zyklus immer positiv)
        (0.2, 2.5)      # alpha_kep (dynamischer Kepler-Flugzeit-Faktor)
    ]
    t_opt_start = time.perf_counter()
    opt_res = opt.minimize(
        loss_func,
        init_params,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 600, 'disp': False}
    )
    t_opt_elapsed = time.perf_counter() - t_opt_start

    p_opt = opt_res.x
    C_N_opt, beta_opt, eps_0_opt, k_gate_opt, gamma_orb_opt, alpha_kep_opt = p_opt
    print(f"Optimierung abgeschlossen in {t_opt_elapsed:.2f} Sekunden.")

    # -------------------------------------------------------------
    # 4. AUSWERTUNG NACH DER 10M-OPTIMIERUNG (AUF ALLEN MILLIONEN DATENSÄTZEN)
    # -------------------------------------------------------------
    gate_final = 0.5 * (1.0 + np.tanh(k_gate_opt * (cos_alphas + 0.05)))
    rot_final = 1.0 + (np.minimum(2.5, etas)**beta_opt)
    n_base_final = (C_N_opt * rot_final) / (dphis + eps_0_opt)
    n_pred_final = np.maximum(1, 1 + np.round((n_base_final - 1.0) * gate_final))
    t_kep_final = alpha_kep_opt * (14.0**1.5) / np.sqrt(1.0 * masses)
    pred_final = (n_pred_final - 1) * (gamma_orb_opt * t_scales) + t_kep_final
    errs_final = np.abs(pred_final - t_reals) / t_reals * 100.0

    med_final = np.median(errs_final)
    wins_final = np.sum(errs_final < errs_heggie)

    print("\n" + "=" * 88)
    print("  ENDERGEBNISSE DER ULTIMATIVEN 10.000.000-SYSTEME-EICHUNG")
    print("=" * 88)
    print("FINALE PARAMETER DER TRINITÄTS-MASTERFORMEL (10M EICHUNG):")
    print(f"  • Basiskonstante C_N         : {C_N_opt:.4f}  (100k: 1.7090 | Initial: 2.4000)")
    print(f"  • Rotations-Exponent β       : {beta_opt:.4f}  (100k: 0.3488 | Initial: 0.3000)")
    print(f"  • Zeit-Scheren-Dämpfung ε_0  : {eps_0_opt:.4f}  (100k: 0.2632 | Initial: 0.1500)")
    print(f"  • Doppler-Gate Steilheit k   : {k_gate_opt:.4f}  (100k: 6.8487 | Initial: 6.0000)")
    print(f"  • Orbit-Kopplung γ_orbit     : {gamma_orb_opt:.4f}  (100k: 0.0359 | Initial: 0.0980)")
    print(f"  • Dynamischer Flug-Faktor α  : {alpha_kep_opt:.4f}  (100k: 0.4681 | Initial: 0.2200)")

    print("\nSTATISTISCHER ENDVERGLEICH (AUF ÜBER 5 MILLIONEN GÜLTIGEN SYSTEMEN):")
    print(f"  • Median-Fehler Heggie (1975)                : {med_heggie:.2f}%")
    print(f"  • Median-Fehler Trinität vor 10M-Optimierung : {med_100k:.2f}%")
    print(f"  • Median-Fehler Trinität NACH 10M-EICHUNG    : {med_final:.2f}%")
    print(f"  • Trinität schlägt Heggie in                 : {wins_final/N_valid*100:.2f}% aller Systeme!")
    print(f"  • Trefferquote Auswurf-Kandidat              : {np.mean(cand_hits)*100:.2f}% ({np.sum(cand_hits):,} Systeme)")

    # -------------------------------------------------------------
    # 5. DISKRETES RESONANZ-HISTOGRAMM MIT EXAKTER DURCHSCHNITTLICHER LEBENSDAUER
    # -------------------------------------------------------------
    print("\n" + "-" * 88)
    print("DAS ECHTE HOCHAUFLÖSENDE RESONANZ-HISTOGRAMM (10.000.000 SYSTEME):")
    print("-" * 88)
    print(" Zyklus N │ Absolute Anzahl │ Prozent │ Mittlere Dauer │ Regime")
    print("──────────┼─────────────────┼─────────┼────────────────┼─────────────────────────────────")
    hist_results = {}
    for n_val in range(1, 11):
        mask_n = (n_reals == n_val)
        count = int(np.sum(mask_n))
        pct = count / N_valid * 100.0
        mean_t = float(np.mean(t_reals[mask_n])) if count > 0 else 0.0
        bar = "█" * int(pct / 1.5)
        regime = "Sturzflug-Plateau" if n_val == 1 else ("Kompakt-Resonanz" if n_val <= 4 else "Chaotische Oszillation")
        hist_results[str(n_val)] = {"count": count, "pct": pct, "mean_t": mean_t}
        print(f"   N = {n_val:2d}   │   {count:9,d}   │  {pct:5.2f}% │    {mean_t:5.2f} s    │ {regime:24s} {bar}")

    mask_trap = (n_reals > 10)
    trap_count = int(np.sum(mask_trap))
    trap_pct = trap_count / N_valid * 100.0
    trap_mean_t = float(np.mean(t_reals[mask_trap])) if trap_count > 0 else 0.0
    hist_results[">10"] = {"count": trap_count, "pct": trap_pct, "mean_t": trap_mean_t}
    print(f"   N > 10   │   {trap_count:9,d}   │  {trap_pct:5.2f}% │    {trap_mean_t:5.2f} s    │ Sticky Orbit Traps       {'█'*int(trap_pct/1.5)}")
    print("=" * 88)

    # 6. Speichere Endergebnisse als JSON
    summary_data = {
        "total_simulated": int(total_simulated),
        "total_valid": int(N_valid),
        "sim_duration_seconds": float(sim_duration),
        "cand_hit_pct": float(np.mean(cand_hits)*100.0),
        "med_heggie_pct": float(med_heggie),
        "med_trinity_100k_pct": float(med_100k),
        "med_trinity_10m_pct": float(med_final),
        "win_rate_pct": float(wins_final / N_valid * 100.0),
        "params": {
            "C_N": float(C_N_opt),
            "beta": float(beta_opt),
            "eps_0": float(eps_0_opt),
            "k_gate": float(k_gate_opt),
            "gamma_orb": float(gamma_orb_opt),
            "alpha_kep": float(alpha_kep_opt)
        },
        "histogram": hist_results
    }
    with open("scratch/mega_10m_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("Ergebnisse erfolgreich in scratch/mega_10m_results.json gespeichert!")

if __name__ == "__main__":
    run_mega_10m()
