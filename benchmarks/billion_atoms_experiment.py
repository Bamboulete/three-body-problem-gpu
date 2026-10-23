"""
DAS 1-MILLIARDE-QUANTENATOM-EXPERIMENT: COULOMB-DREIKÖRPERPROBLEM AUF DER GPU
Vollständige Simulation von 1.000.000.000 doppelt angeregten Helium-Atomen auf der NVIDIA RTX 4070.

Wissenschaftliche Fragestellung:
Entsteht auf atomarer Ebene (Coulomb-Abstoßung +1/r_12) bei 1 Milliarde Systemen
dieselbe universelle Phasenraum-Topologie und derselbe Phase-2 Poisson-Zerfall
wie bei 1 Milliarde gravitativer Sterne?

Hardware: NVIDIA GeForce RTX 4070 (12 GB VRAM)
Setup:
- 250 Batches à 4.000.000 Helium-Systeme = 1.000.000.000 Systeme
- Symplektischer Leapfrog-Integrator in atomaren Einheiten (dt = 0.015 a.u. ~ 0.36 attoseconds)
- 500 x 500 Phasenraum-Dichtematrix: Coulomb Zeit-Schere ΔΦ_Coulomb vs. Doppler-Tor cos(α)
- Exakte Messung der Symmetrie von Elektron 1 vs. Elektron 2 (Quanten-Massen-Zentrifuge)
- Fit der universellen Zerfallskonstante λ_Coulomb
"""

import time
import math
import sys
import os
import json
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Optimierter CUDA Kernel für 1 Milliarde Atome mit NVIDIA Fast-Math
cuda_kernel_source = r'''
extern "C" __global__
void simulate_helium_billion(
    const float* __restrict__ r1_in,
    const float* __restrict__ v1_in,
    const float* __restrict__ r2_in,
    const float* __restrict__ v2_in,
    float* __restrict__ t_ion_out,
    int* __restrict__ ejected_out,
    const int num_systems,
    const float dt,
    const float t_max,
    const float r_escape
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_systems) return;

    const float Z = 2.0f;
    const float inv_Mnuc = 1.0f / 7294.3f;
    const float eps2 = 0.0004f; // Softening eps = 0.02 a.u.

    float r1x = r1_in[idx * 3 + 0], r1y = r1_in[idx * 3 + 1], r1z = r1_in[idx * 3 + 2];
    float v1x = v1_in[idx * 3 + 0], v1y = v1_in[idx * 3 + 1], v1z = v1_in[idx * 3 + 2];

    float r2x = r2_in[idx * 3 + 0], r2y = r2_in[idx * 3 + 1], r2z = r2_in[idx * 3 + 2];
    float v2x = v2_in[idx * 3 + 0], v2y = v2_in[idx * 3 + 1], v2z = v2_in[idx * 3 + 2];

    float r0x = 0.0f, r0y = 0.0f, r0z = 0.0f;
    float v0x = 0.0f, v0y = 0.0f, v0z = 0.0f;

    float t = 0.0f;
    const int max_steps = (int)(t_max / dt);
    const float dt2_half = 0.5f * dt * dt;

    int ejected_elec = 0;
    float t_ion = t_max;

    // Start-Kräfte
    float d10x = r1x - r0x, d10y = r1y - r0y, d10z = r1z - r0z;
    float inv_d10 = rsqrtf(d10x * d10x + d10y * d10y + d10z * d10z + eps2);
    float inv_d10_3 = inv_d10 * inv_d10 * inv_d10;

    float d20x = r2x - r0x, d20y = r2y - r0y, d20z = r2z - r0z;
    float inv_d20 = rsqrtf(d20x * d20x + d20y * d20y + d20z * d20z + eps2);
    float inv_d20_3 = inv_d20 * inv_d20 * inv_d20;

    float d12x = r1x - r2x, d12y = r1y - r2y, d12z = r1z - r2z;
    float inv_d12 = rsqrtf(d12x * d12x + d12y * d12y + d12z * d12z + eps2);
    float inv_d12_3 = inv_d12 * inv_d12 * inv_d12;

    float a1x = -Z * d10x * inv_d10_3 + d12x * inv_d12_3;
    float a1y = -Z * d10y * inv_d10_3 + d12y * inv_d12_3;
    float a1z = -Z * d10z * inv_d10_3 + d12z * inv_d12_3;

    float a2x = -Z * d20x * inv_d20_3 - d12x * inv_d12_3;
    float a2y = -Z * d20y * inv_d20_3 - d12y * inv_d12_3;
    float a2z = -Z * d20z * inv_d20_3 - d12z * inv_d12_3;

    float a0x = (Z * d10x * inv_d10_3 + Z * d20x * inv_d20_3) * inv_Mnuc;
    float a0y = (Z * d10y * inv_d10_3 + Z * d20y * inv_d20_3) * inv_Mnuc;
    float a0z = (Z * d10z * inv_d10_3 + Z * d20z * inv_d20_3) * inv_Mnuc;

    #pragma unroll 4
    for (int step = 0; step < max_steps; ++step) {
        r0x += v0x * dt + a0x * dt2_half;
        r0y += v0y * dt + a0y * dt2_half;
        r0z += v0z * dt + a0z * dt2_half;

        r1x += v1x * dt + a1x * dt2_half;
        r1y += v1y * dt + a1y * dt2_half;
        r1z += v1z * dt + a1z * dt2_half;

        r2x += v2x * dt + a2x * dt2_half;
        r2y += v2y * dt + a2y * dt2_half;
        r2z += v2z * dt + a2z * dt2_half;

        d10x = r1x - r0x; d10y = r1y - r0y; d10z = r1z - r0z;
        inv_d10 = rsqrtf(d10x * d10x + d10y * d10y + d10z * d10z + eps2);
        inv_d10_3 = inv_d10 * inv_d10 * inv_d10;

        d20x = r2x - r0x; d20y = r2y - r0y; d20z = r2z - r0z;
        inv_d20 = rsqrtf(d20x * d20x + d20y * d20y + d20z * d20z + eps2);
        inv_d20_3 = inv_d20 * inv_d20 * inv_d20;

        d12x = r1x - r2x; d12y = r1y - r2y; d12z = r1z - r2z;
        inv_d12 = rsqrtf(d12x * d12x + d12y * d12y + d12z * d12z + eps2);
        inv_d12_3 = inv_d12 * inv_d12 * inv_d12;

        float new_a1x = -Z * d10x * inv_d10_3 + d12x * inv_d12_3;
        float new_a1y = -Z * d10y * inv_d10_3 + d12y * inv_d12_3;
        float new_a1z = -Z * d10z * inv_d10_3 + d12z * inv_d12_3;

        float new_a2x = -Z * d20x * inv_d20_3 - d12x * inv_d12_3;
        float new_a2y = -Z * d20y * inv_d20_3 - d12x * inv_d12_3;
        float new_a2z = -Z * d20z * inv_d20_3 - d12z * inv_d12_3;

        float new_a0x = (Z * d10x * inv_d10_3 + Z * d20x * inv_d20_3) * inv_Mnuc;
        float new_a0y = (Z * d10y * inv_d10_3 + Z * d20y * inv_d20_3) * inv_Mnuc;
        float new_a0z = (Z * d10z * inv_d10_3 + Z * d20z * inv_d20_3) * inv_Mnuc;

        v0x += 0.5f * (a0x + new_a0x) * dt;
        v0y += 0.5f * (a0y + new_a0y) * dt;
        v0z += 0.5f * (a0z + new_a0z) * dt;

        v1x += 0.5f * (a1x + new_a1x) * dt;
        v1y += 0.5f * (a1y + new_a1y) * dt;
        v1z += 0.5f * (a1z + new_a1z) * dt;

        v2x += 0.5f * (a2x + new_a2x) * dt;
        v2y += 0.5f * (a2y + new_a2y) * dt;
        v2z += 0.5f * (a2z + new_a2z) * dt;

        a0x = new_a0x; a0y = new_a0y; a0z = new_a0z;
        a1x = new_a1x; a1y = new_a1y; a1z = new_a1z;
        a2x = new_a2x; a2y = new_a2y; a2z = new_a2z;

        t += dt;

        // Autoionisations-Prüfung alle 16 Schritte
        if ((step & 15) == 0) {
            float dist1 = 1.0f / inv_d10;
            float dist2 = 1.0f / inv_d20;

            if (dist1 > r_escape) {
                float v1_sq = v1x * v1x + v1y * v1y + v1z * v1z;
                if (0.5f * v1_sq - 1.0f * inv_d10 > 0.0f) {
                    ejected_elec = 1;
                    t_ion = t;
                    break;
                }
            } else if (dist2 > r_escape) {
                float v2_sq = v2x * v2x + v2y * v2y + v2z * v2z;
                if (0.5f * v2_sq - 1.0f * inv_d20 > 0.0f) {
                    ejected_elec = 2;
                    t_ion = t;
                    break;
                }
            }
        }
    }

    t_ion_out[idx] = t_ion;
    ejected_out[idx] = ejected_elec;
}
'''

def run_billion_atoms_experiment():
    TOTAL_TARGET = 1_000_000_000
    BATCH_SIZE = 4_000_000
    NUM_BATCHES = TOTAL_TARGET // BATCH_SIZE # 250 Batches

    print("=" * 95)
    print("  DAS 1-MILLIARDE-QUANTENATOM-EXPERIMENT: COULOMB-DREIKÖRPERPROBLEM  ".center(95))
    print("  HARDWARE: NVIDIA GEFORCE RTX 4070 (12 GB VRAM) - 1.000.000.000 HELIUM-SYSTEME  ".center(95))
    print("=" * 95)
    print(f"  Geplante Systeme   : {TOTAL_TARGET:,} Helium-Atome (Kern Z=+2, 2 Elektronen)")
    print(f"  Etappen            : {NUM_BATCHES} Batches à {BATCH_SIZE:,} Systeme")
    print(f"  Zeitschritt dt     : 0.015 a.u. (~ 0.36 Attosekunden)")
    print(f"  Simulierte Zeit    : 200.0 a.u. (~ 4.8 Femtosekunden)")
    print("=" * 95)

    print("Kompiliere spezialisierten CUDA C++ Fast-Math Kernel...")
    module = cp.RawModule(code=cuda_kernel_source, options=('--use_fast_math',))
    kernel = module.get_function('simulate_helium_billion')

    # Globale GPU-Akkumulatoren
    # 500x500 2D-Phasenraum Matrix:
    # X: Coulomb Zeit-Schere ΔΦ in [0.0, 0.8] -> 500 Bins
    # Y: Doppler-Tor cos(α) in [-1.0, 1.0] -> 500 Bins
    GRID_RES = 500
    phase_space_grid_gpu = cp.zeros((GRID_RES, GRID_RES), dtype=cp.int64)

    # Zerfalls-Histogramm (50 Bins von t = 10 bis 200 a.u.)
    NUM_TBINS = 50
    t_hist_gpu = cp.zeros(NUM_TBINS, dtype=cp.int64)

    # Globale Zähler
    total_simulated = 0
    total_ionized = 0
    total_bound = 0
    total_e1_ejected = 0
    total_e2_ejected = 0

    block_size = 256
    grid_size = (BATCH_SIZE + block_size - 1) // block_size

    # Puffer im GPU VRAM vorab allozieren
    r1_gpu = cp.empty((BATCH_SIZE, 3), dtype=cp.float32)
    v1_gpu = cp.empty((BATCH_SIZE, 3), dtype=cp.float32)
    r2_gpu = cp.empty((BATCH_SIZE, 3), dtype=cp.float32)
    v2_gpu = cp.empty((BATCH_SIZE, 3), dtype=cp.float32)
    t_ion_gpu = cp.zeros(BATCH_SIZE, dtype=cp.float32)
    ejected_gpu = cp.zeros(BATCH_SIZE, dtype=cp.int32)

    dt = np.float32(0.015)
    t_max = np.float32(200.0)
    r_escape = np.float32(12.0)

    start_time = time.perf_counter()
    print("Start der 1-Milliarde-Atom-Simulation... Rechnet auf GPU...\n")

    for b in range(1, NUM_BATCHES + 1):
        t_batch_start = time.perf_counter()

        # 1. Erzeuge 4.000.000 Helium-Systeme direkt im VRAM
        cp.random.seed(300000 + b * 1337)
        r1_mag = cp.random.uniform(1.0, 1.5, BATCH_SIZE, dtype=cp.float32)
        r2_mag = cp.random.uniform(1.5, 3.5, BATCH_SIZE, dtype=cp.float32)
        
        # Coulomb Zeit-Schere ΔΦ = |r2 - r1| / (r2 + r1)
        delta_phi = cp.abs(r2_mag - r1_mag) / (r2_mag + r1_mag)

        # Doppler-Tor cos(α)
        cos_alpha = cp.random.uniform(-1.0, 1.0, BATCH_SIZE, dtype=cp.float32)
        alpha_rad = cp.arccos(cos_alpha)

        theta1 = cp.random.uniform(0, 2*np.pi, BATCH_SIZE, dtype=cp.float32)
        r1_gpu[:, 0] = r1_mag * cp.cos(theta1)
        r1_gpu[:, 1] = r1_mag * cp.sin(theta1)
        r1_gpu[:, 2] = 0.0

        v1_mag = cp.sqrt(2.0 / r1_mag)
        v1_gpu[:, 0] = -v1_mag * cp.sin(theta1)
        v1_gpu[:, 1] = v1_mag * cp.cos(theta1)
        v1_gpu[:, 2] = 0.0

        theta2 = cp.random.uniform(0, 2*np.pi, BATCH_SIZE, dtype=cp.float32)
        r2_gpu[:, 0] = r2_mag * cp.cos(theta2)
        r2_gpu[:, 1] = r2_mag * cp.sin(theta2)
        r2_gpu[:, 2] = 0.0

        v2_mag = cp.sqrt(1.0 / r2_mag)
        v2_gpu[:, 0] = -v2_mag * cp.sin(theta2) * cp.cos(alpha_rad)
        v2_gpu[:, 1] = v2_mag * cp.cos(theta2) * cp.cos(alpha_rad)
        v2_gpu[:, 2] = v2_mag * cp.sin(alpha_rad)

        # 2. Kernel ausführen
        kernel(
            (grid_size,), (block_size,),
            (r1_gpu, v1_gpu, r2_gpu, v2_gpu, t_ion_gpu, ejected_gpu,
             np.int32(BATCH_SIZE), dt, t_max, r_escape)
        )

        # 3. Zählung und Akkumulation auf GPU
        is_ionized = ejected_gpu > 0
        n_ionized = int(cp.sum(is_ionized))
        n_e1 = int(cp.sum(ejected_gpu == 1))
        n_e2 = int(cp.sum(ejected_gpu == 2))

        total_simulated += BATCH_SIZE
        total_ionized += n_ionized
        total_bound += (BATCH_SIZE - n_ionized)
        total_e1_ejected += n_e1
        total_e2_ejected += n_e2

        # 4. 2D Phasenraum-Dichtematrix für zerfallene Atome akkumulieren
        # X: Delta_Phi in [0.0, 0.8] -> 500 Bins
        # Y: cos_alpha in [-1.0, 1.0] -> 500 Bins
        ionized_indices = cp.where(is_ionized)[0]
        if len(ionized_indices) > 0:
            dphi_ion = delta_phi[ionized_indices]
            cos_ion = cos_alpha[ionized_indices]

            bin_x = cp.clip((dphi_ion / 0.8 * GRID_RES).astype(cp.int32), 0, GRID_RES - 1)
            bin_y = cp.clip(((cos_ion + 1.0) / 2.0 * GRID_RES).astype(cp.int32), 0, GRID_RES - 1)

            # Flat Index für 2D-Histogramm
            flat_bins = bin_y * GRID_RES + bin_x
            counts = cp.bincount(flat_bins, minlength=GRID_RES * GRID_RES)
            phase_space_grid_gpu += counts.reshape((GRID_RES, GRID_RES))

            # Zeit-Histogramm
            t_ion_vals = t_ion_gpu[ionized_indices]
            t_bins = cp.clip(((t_ion_vals - 10.0) / (t_max - 10.0) * NUM_TBINS).astype(cp.int32), 0, NUM_TBINS - 1)
            t_counts = cp.bincount(t_bins, minlength=NUM_TBINS)
            t_hist_gpu += t_counts

        t_batch_end = time.perf_counter()
        batch_fps = BATCH_SIZE / (t_batch_end - t_batch_start)

        # Fortschrittsausgabe alle 25 Batches
        if b % 25 == 0 or b == NUM_BATCHES:
            elapsed = t_batch_end - start_time
            overall_fps = total_simulated / elapsed
            progress_pct = (b / NUM_BATCHES) * 100.0
            print(f"Batch {b:3d}/{NUM_BATCHES} ({progress_pct:5.1f}%) | "
                  f"Gesamt: {total_simulated:11,d} Atome | "
                  f"Ionisiert: {total_ionized:10,d} ({total_ionized/total_simulated*100:5.2f}%) | "
                  f"Speed: {overall_fps:,.0f} Sys/s | "
                  f"Zeit: {elapsed:5.1f}s")

    total_time = time.perf_counter() - start_time
    overall_throughput = total_simulated / total_time

    print("\n" + "=" * 95)
    print("  1-MILLIARDE-ATOM-EXPERIMENT ERFOLGREICH VOLLENDET!  ".center(95))
    print("=" * 95)
    print(f"  Hardware                 : NVIDIA GeForce RTX 4070 (12 GB VRAM)")
    print(f"  Gesamtsysteme berechnet  : {total_simulated:,} Helium-Atome")
    print(f"  Gesamte Rechenzeit       : {total_time:.2f} Sekunden ({total_time/60.0:.2f} Minuten)")
    print(f"  Schnitt-Durchsatz        : {overall_throughput:,.0f} Atome / Sekunde")
    print(f"  Autoionisiert (Zerfallen): {total_ionized:,} ({total_ionized/total_simulated*100:.2f} %)")
    print(f"  Metastabil gebunden      : {total_bound:,} ({total_bound/total_simulated*100:.2f} %)")
    print(f"  Elektron 2 ausgeworfen   : {total_e2_ejected:,} ({total_e2_ejected/total_ionized*100:.2f} %)")
    print(f"  Elektron 1 ausgeworfen   : {total_e1_ejected:,} ({total_e1_ejected/total_ionized*100:.2f} %)")
    print("-" * 95)

    # 5. Phasenraum-Matrix auf CPU holen und Bild rendern
    grid_cpu = cp.asnumpy(phase_space_grid_gpu)
    t_hist_cpu = cp.asnumpy(t_hist_gpu)

    # 6. Fit der Zerfallskonstante λ
    t_bin_centers = np.linspace(10.0 + (t_max - 10.0)/(2*NUM_TBINS), t_max - (t_max - 10.0)/(2*NUM_TBINS), NUM_TBINS)
    # Exponentieller Fit für Phase 2 (t > 30 a.u.)
    fit_mask = t_bin_centers >= 30.0
    t_fit = t_bin_centers[fit_mask]
    counts_fit = t_hist_cpu[fit_mask]
    ln_counts = np.log(np.maximum(1, counts_fit))

    poly = np.polyfit(t_fit, ln_counts, 1)
    lambda_1b_coulomb = -poly[0]
    r2_1b = 1.0 - (np.sum((ln_counts - np.polyval(poly, t_fit))**2) / np.sum((ln_counts - np.mean(ln_counts))**2))
    lambda_1b_orb = lambda_1b_coulomb * 12.57 # pro atomarem Orbit

    print(f"  Gemessene 1-Milliarde-Zerfallskonstante: λ_Coulomb = {lambda_1b_coulomb:.5f} a.u.^-1")
    print(f"  Zerfallsrate pro atomarem Umlauf       : λ_orb = {lambda_1b_orb:.4f} pro Orbit (R² = {r2_1b:.4f})")
    print(f"  Kosmischer 1-Milliarde-Sterne-Vergleich: λ_grav = 0.0567 pro Orbit (R² = 0.9973)")
    print(f"  Relative Übereinstimmung Kosmos vs Atom: {100.0 - abs(lambda_1b_orb - 0.0567)/0.0567*100.0:.2f} %!")
    print("=" * 95)

    # 7. Render 500x500 Heatmap
    heatmap_path = os.path.join("assets", "phase_space_density_1b_atoms.png")
    os.makedirs("assets", exist_ok=True)
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.style.use('dark_background')
    
    # Log-Skala für Density
    log_grid = np.log10(grid_cpu + 1.0)
    plt.imshow(
        log_grid,
        origin='lower',
        extent=[0.0, 0.8, -1.0, 1.0],
        aspect='auto',
        cmap='plasma'
    )
    plt.colorbar(label='Log10(Ereignisse pro Phasenraum-Zelle)')
    plt.xlabel('Coulomb Zeit-Schere ΔΦ_Coulomb = |r2 - r1| / (r2 + r1)')
    plt.ylabel('Doppler-Tor cos(α) = cos(∠(v1, v2))')
    plt.title('1.000.000.000 Helium-Atome: Phasenraum-Dichte (Coulomb 3-Körper-Zerfall)')
    
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"\nPhasenraum-Dichtekarte (500x500) erfolgreich gerendert:\n{heatmap_path}\n")

    # Ergebnisse speichern
    res_path = os.path.join("benchmarks", "billion_atoms_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_simulated": total_simulated,
            "total_time_sec": total_time,
            "throughput_sys_per_sec": overall_throughput,
            "total_ionized": total_ionized,
            "total_bound": total_bound,
            "total_e1_ejected": total_e1_ejected,
            "total_e2_ejected": total_e2_ejected,
            "lambda_coulomb_au": float(lambda_1b_coulomb),
            "lambda_per_orbit": float(lambda_1b_orb),
            "r_squared": float(r2_1b),
            "lambda_grav_per_orbit": 0.0567
        }, f, indent=2)
    print(f"Vollständige 1-Milliarde-Statistik gespeichert in:\n{res_path}\n")

if __name__ == "__main__":
    run_billion_atoms_experiment()
