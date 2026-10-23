"""
Trinitäts-Topologie: Erprobung der Multi-Eigenzeit-Hypothese
Prüft, ob die relative Zeitverzerrung (Eigenzeiten tau_a, tau_b, tau_c)
und der Zeit-Gradient den Phasen-Abriss und Auswurf bei t=63.2s exakt erklärt.
"""

import numpy as np
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_proper_time_analysis(name, m, r, v, G=1.0, eps=0.05, t_max=75.0, dt=0.002, c_eff=10.0):
    r = np.array(r, dtype=float)
    v = np.array(v, dtype=float)
    m = np.array(m, dtype=float)
    
    t = 0.0
    steps = int(t_max / dt)
    
    # Akkumulierte Eigenzeiten für jeden Körper
    tau = np.zeros(3)
    
    # Verlaufsprotokoll
    history = []
    
    def accel(pos):
        a = np.zeros_like(pos)
        for i in range(3):
            for j in range(i+1, 3):
                dr = pos[j] - pos[i]
                d2 = np.sum(dr**2) + eps**2
                inv3 = G / (d2 * np.sqrt(d2))
                a[i] += inv3 * m[j] * dr
                a[j] -= inv3 * m[i] * dr
        return a

    def get_time_rates(pos, vel):
        """Berechnet d(tau_i)/dt für jeden der drei Körper."""
        rates = np.zeros(3)
        c2 = c_eff ** 2
        for i in range(3):
            v2 = np.sum(vel[i]**2)
            # Gravitationspotential am Ort von Körper i
            phi_i = 0.0
            for j in range(3):
                if i != j:
                    dist = np.sqrt(np.sum((pos[j] - pos[i])**2) + eps**2)
                    phi_i -= (G * m[j]) / dist
            
            # Relativistischer / geometrischer Dilatationsfaktor
            # dtau = sqrt(1 - v^2/c^2 + 2*phi/c^2)
            factor = 1.0 - (v2 / c2) + (2.0 * phi_i / c2)
            rates[i] = np.sqrt(max(0.001, factor))
        return rates

    ejected = False
    eject_time = None
    eject_idx = None
    
    # Meilensteine erfassen
    log_milestones = [10.0, 30.0, 50.0, 60.0, 62.0, 63.2, 65.0]

    print(f"\n=================================================================")
    print(f"  Multi-Eigenzeit-Analyse für [{name}]")
    print(f"  Effektive Lichtgeschwindigkeit c_eff = {c_eff}")
    print(f"=================================================================")

    for s in range(steps):
        # 1. Zeit-Flussraten berechnen
        rates = get_time_rates(r, v)
        tau += rates * dt
        t += dt
        
        # RK4 Integration
        a1 = accel(r)
        k1_r = v
        k1_v = a1
        
        a2 = accel(r + 0.5*dt*k1_r)
        k2_r = v + 0.5*dt*k1_v
        k2_v = a2
        
        a3 = accel(r + 0.5*dt*k2_r)
        k3_r = v + 0.5*dt*k2_v
        k3_v = a3
        
        a4 = accel(r + dt*k3_r)
        k4_r = v + dt*k3_v
        k4_v = a4
        
        r += (dt/6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v += (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        cm = np.sum(r * m[:, None], axis=0) / np.sum(m)
        dists_cm = np.linalg.norm(r - cm, axis=1)

        # Auswurf-Erkennung
        if np.max(dists_cm) > 15.0 and not ejected:
            ejected = True
            eject_time = t
            eject_idx = np.argmax(dists_cm)

        # Logging zu Schlüsselzeitpunkten
        for ms in log_milestones:
            if abs(t - ms) < dt * 0.6:
                diff_ab = tau[0] - tau[1]
                diff_pair_c = np.mean([tau[0], tau[1]]) - tau[2]
                ratio_c = rates[2] / max(0.001, np.mean([rates[0], rates[1]]))
                print(f"t = {t:5.1f}s | τ_A = {tau[0]:6.3f}s, τ_B = {tau[1]:6.3f}s, τ_C = {tau[2]:6.3f}s | Zeit-Verhältnis dτ_C/dτ_Paar = {ratio_c:5.2f}")

    if ejected:
        print(f"\n>>> ERGEBNIS:")
        print(f" -> Auswurf von Stern {chr(65+eject_idx)} (Masse {m[eject_idx]}) bei t = {eject_time:.2f} s")
        print(f" -> Akkumulierte Eigenzeit-Differenz bei Auswurf: Δτ = {tau[eject_idx] - np.mean(np.delete(tau, eject_idx)):.4f} s")
    else:
        print(f"\n>>> Kein Auswurf festgestellt bis t = {t_max}s.")

if __name__ == "__main__":
    # 1. Figure-8 (Symmetrisch - alle sollten identische Eigenzeit haben!)
    x1, y1 = -0.97000436, 0.24308753
    vx1, vy1 = 0.46620531, 0.43236573
    m_fig = [1.0, 1.0, 1.0]
    r_fig = [[x1, y1, 0], [-x1, -y1, 0], [0, 0, 0]]
    v_fig = [[vx1, vy1, 0], [vx1, vy1, 0], [-2*vx1, -2*vy1, 0]]
    run_proper_time_analysis("Figure-8 Choreographie", m_fig, r_fig, v_fig, t_max=15.0)

    # 2. Trisolaris Chaos (Der Härtetest für t = 63.2s!)
    m_tri = [1.4, 1.1, 0.9]
    r_tri = [[-1.8, 0.5, 0.2], [1.5, -0.8, -0.3], [0.2, 1.6, 0.1]]
    v_tri = [[0.1, -0.65, 0.1], [-0.4, 0.5, -0.1], [0.35, -0.15, 0.05]]
    run_proper_time_analysis("Trisolaris Chaos", m_tri, r_tri, v_tri, t_max=70.0)
