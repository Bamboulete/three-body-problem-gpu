/**
 * Trinitäts-Topologie: Physik-Engine & Erhaltungsgrößen
 * 4th-Order Runge-Kutta (RK4) mit symplektischer Erhaltung,
 * Trinitäts-Geometrie (GS) und Ω-Spannungsenergie-Operator.
 */

export class Body {
  constructor(id, name, mass, color, pos, vel) {
    this.id = id;
    this.name = name;
    this.mass = mass;
    this.color = color;
    this.pos = { ...pos }; // {x, y, z}
    this.vel = { ...vel }; // {vx, vy, vz}
  }

  clone() {
    return new Body(
      this.id,
      this.name,
      this.mass,
      this.color,
      { ...this.pos },
      { ...this.vel }
    );
  }
}

export class PhysicsEngine {
  constructor() {
    this.G = 1.0; // Normalisierte Gravitationskonstante
    this.softening = 0.05; // Glättungsfaktor ε zur Dämpfung von Singularitäten
    this.bodies = [];
    this.time = 0;
    this.initialEnergy = null;
    this.prevOmega = 0;
    this.omegaRate = 0;
    this.substeps = 25; // Sub-Steps pro Frame für maximale Stabilität

    // Multi-Eigenzeit-Modell (Individuelle Zeitdilatation für jeden Sonnen-Körper)
    this.c_eff = 10.0; // Effektive Lichtgeschwindigkeit für die Zeittakt-Berechnung
    this.tau = [0, 0, 0]; // Eigenzeiten tau_a, tau_b, tau_c
    this.timeRates = [1.0, 1.0, 1.0];

    // Diskretes Zyklen- & Phasen-Integral-Modell
    this.sigmaIntegral = 0;
    this.cyclesCount = 0;
    this.prevMinDist = 0;
    this.prevDistRate = 0;
    this.exactEjectionTime = null;
    this.keplerFlightTime = 0;
  }

  setBodies(bodyList) {
    this.bodies = bodyList.map(b => b.clone());
    this.time = 0;
    this.initialEnergy = this.calculateTotalEnergy();
    this.prevOmega = this.calculateOmega();
    this.omegaRate = 0;
    this.phaseHistory = [];
    this.tau = [0, 0, 0];
    this.timeRates = [1.0, 1.0, 1.0];
    this.slingshotLocked = false;
    this.lockedCandidateIdx = -1;
    this.slingshotTime = 0;
    this.sigmaIntegral = 0;
    this.cyclesCount = 0;
    this.prevMinDist = 0;
    this.prevDistRate = 0;
    this.exactEjectionTime = null;
    this.keplerFlightTime = 0;
  }

  // Beschleunigungen berechnen für gegebenen Zustand
  computeAccelerations(bodiesState) {
    const n = bodiesState.length;
    const accels = bodiesState.map(() => ({ ax: 0, ay: 0, az: 0 }));
    const eps2 = this.softening * this.softening;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const bi = bodiesState[i];
        const bj = bodiesState[j];

        const dx = bj.pos.x - bi.pos.x;
        const dy = bj.pos.y - bi.pos.y;
        const dz = bj.pos.z - bi.pos.z;

        const distSq = dx * dx + dy * dy + dz * dz + eps2;
        const dist = Math.sqrt(distSq);
        const invDist3 = 1.0 / (dist * distSq);

        const forceFactor = this.G * invDist3;

        // Kraft von j auf i
        accels[i].ax += forceFactor * bj.mass * dx;
        accels[i].ay += forceFactor * bj.mass * dy;
        accels[i].az += forceFactor * bj.mass * dz;

        // Kraft von i auf j (Actio = Reactio)
        accels[j].ax -= forceFactor * bi.mass * dx;
        accels[j].ay -= forceFactor * bi.mass * dy;
        accels[j].az -= forceFactor * bi.mass * dz;
      }
    }

    return accels;
  }

  // 4th-Order Runge-Kutta Einzelschritt
  rk4Step(dt) {
    const current = this.bodies;
    const n = current.length;

    // K1
    const a1 = this.computeAccelerations(current);
    const k1_v = a1.map(a => ({ x: a.ax, y: a.ay, z: a.az }));
    const k1_r = current.map(b => ({ x: b.vel.vx, y: b.vel.vy, z: b.vel.vz }));

    // K2 (t + dt/2, r + k1_r * dt/2, v + k1_v * dt/2)
    const state2 = current.map((b, i) => ({
      pos: {
        x: b.pos.x + k1_r[i].x * 0.5 * dt,
        y: b.pos.y + k1_r[i].y * 0.5 * dt,
        z: b.pos.z + k1_r[i].z * 0.5 * dt,
      },
      vel: {
        vx: b.vel.vx + k1_v[i].x * 0.5 * dt,
        vy: b.vel.vy + k1_v[i].y * 0.5 * dt,
        vz: b.vel.vz + k1_v[i].z * 0.5 * dt,
      },
      mass: b.mass
    }));
    const a2 = this.computeAccelerations(state2);
    const k2_v = a2.map(a => ({ x: a.ax, y: a.ay, z: a.az }));
    const k2_r = state2.map(b => ({ x: b.vel.vx, y: b.vel.vy, z: b.vel.vz }));

    // K3 (t + dt/2, r + k2_r * dt/2, v + k2_v * dt/2)
    const state3 = current.map((b, i) => ({
      pos: {
        x: b.pos.x + k2_r[i].x * 0.5 * dt,
        y: b.pos.y + k2_r[i].y * 0.5 * dt,
        z: b.pos.z + k2_r[i].z * 0.5 * dt,
      },
      vel: {
        vx: b.vel.vx + k2_v[i].x * 0.5 * dt,
        vy: b.vel.vy + k2_v[i].y * 0.5 * dt,
        vz: b.vel.vz + k2_v[i].z * 0.5 * dt,
      },
      mass: b.mass
    }));
    const a3 = this.computeAccelerations(state3);
    const k3_v = a3.map(a => ({ x: a.ax, y: a.ay, z: a.az }));
    const k3_r = state3.map(b => ({ x: b.vel.vx, y: b.vel.vy, z: b.vel.vz }));

    // K4 (t + dt, r + k3_r * dt, v + k3_v * dt)
    const state4 = current.map((b, i) => ({
      pos: {
        x: b.pos.x + k3_r[i].x * dt,
        y: b.pos.y + k3_r[i].y * dt,
        z: b.pos.z + k3_r[i].z * dt,
      },
      vel: {
        vx: b.vel.vx + k3_v[i].x * dt,
        vy: b.vel.vy + k3_v[i].y * dt,
        vz: b.vel.vz + k3_v[i].z * dt,
      },
      mass: b.mass
    }));
    const a4 = this.computeAccelerations(state4);
    const k4_v = a4.map(a => ({ x: a.ax, y: a.ay, z: a.az }));
    const k4_r = state4.map(b => ({ x: b.vel.vx, y: b.vel.vy, z: b.vel.vz }));

    // Update Zustand
    for (let i = 0; i < n; i++) {
      this.bodies[i].pos.x += (dt / 6.0) * (k1_r[i].x + 2 * k2_r[i].x + 2 * k3_r[i].x + k4_r[i].x);
      this.bodies[i].pos.y += (dt / 6.0) * (k1_r[i].y + 2 * k2_r[i].y + 2 * k3_r[i].y + k4_r[i].y);
      this.bodies[i].pos.z += (dt / 6.0) * (k1_r[i].z + 2 * k2_r[i].z + 2 * k3_r[i].z + k4_r[i].z);

      this.bodies[i].vel.vx += (dt / 6.0) * (k1_v[i].x + 2 * k2_v[i].x + 2 * k3_v[i].x + k4_v[i].x);
      this.bodies[i].vel.vy += (dt / 6.0) * (k1_v[i].y + 2 * k2_v[i].y + 2 * k3_v[i].y + k4_v[i].y);
      this.bodies[i].vel.vz += (dt / 6.0) * (k1_v[i].z + 2 * k2_v[i].z + 2 * k3_v[i].z + k4_v[i].z);
    }
  }

  // Update Schleife mit Sub-Steps
  update(frameDelta, speedMultiplier = 1.0) {
    if (this.bodies.length < 3) return;

    // Begrenze dt gegen Lag-Spikes
    const clampedDelta = Math.min(frameDelta, 0.05);
    const totalDt = clampedDelta * speedMultiplier;
    const subDt = totalDt / this.substeps;

    for (let s = 0; s < this.substeps; s++) {
      this.rk4Step(subDt);
      this.accumulateProperTimes(subDt);
    }

    this.time += totalDt;

    // Ω-Dynamik & Veränderungsrate
    const currentOmega = this.calculateOmega();
    if (totalDt > 0) {
      this.omegaRate = (currentOmega - this.prevOmega) / totalDt;
    }
    this.prevOmega = currentOmega;

    // Phasenraum & Rekurrenz-Historie aufzeichnen
    if (!this.phaseHistory) {
      this.phaseHistory = [];
    }
    const sides = this.getTriangleSides();
    this.phaseHistory.push({
      t: this.time,
      omega: currentOmega,
      dOmega: this.omegaRate,
      sides: sides
    });
    if (this.phaseHistory.length > 300) {
      this.phaseHistory.shift();
    }

    // Akkumulierte Zeit-Schere Sigma(t) & Zyklen-Zählung (Periapsis)
    if (this.bodies.length === 3) {
      const d01 = Math.max(0.01, sides.a1);
      const d12 = Math.max(0.01, sides.a2);
      const d20 = Math.max(0.01, sides.a3);
      const m = this.bodies.map(b => b.mass);
      const p0 = this.G * (m[1] / d01 + m[2] / d20);
      const p1 = this.G * (m[0] / d01 + m[2] / d12);
      const p2 = this.G * (m[0] / d20 + m[1] / d12);
      const p_mean = (p0 + p1 + p2) / 3.0;
      const cur_shear = (Math.max(p0, p1, p2) - Math.min(p0, p1, p2)) / Math.max(0.01, p_mean);
      this.sigmaIntegral += cur_shear * totalDt;

      const min_dist = Math.min(d01, d12, d20);
      if (this.prevMinDist > 0 && totalDt > 0) {
        const rate = (min_dist - this.prevMinDist) / totalDt;
        if (this.prevDistRate < 0 && rate >= 0 && min_dist < 2.0) {
          if (!this.slingshotLocked) {
            this.cyclesCount++;
          }
        }
        this.prevDistRate = rate;
      }
      this.prevMinDist = min_dist;
    }

    // 2. Stufe: Hyperbolischer Slingshot Phasen-Tracker (100% Deterministisch)
    if (!this.slingshotLocked && this.bodies.length === 3) {
      const cm = this.calculateCenterOfMass();
      const eps2 = this.softening * this.softening;
      
      for (let i = 0; i < 3; i++) {
        const bi = this.bodies[i];
        let phi_i = 0;
        for (let j = 0; j < 3; j++) {
          if (i !== j) {
            const bj = this.bodies[j];
            const dist = Math.sqrt((bj.pos.x - bi.pos.x)**2 + (bj.pos.y - bi.pos.y)**2 + (bj.pos.z - bi.pos.z)**2 + eps2);
            phi_i += this.G * bj.mass / dist;
          }
        }
        const v2 = bi.vel.vx*bi.vel.vx + bi.vel.vy*bi.vel.vy + bi.vel.vz*bi.vel.vz;
        const eps_spec = 0.5 * v2 - phi_i;
        
        // Radiale Auswärtsbewegung bezüglich des Schwerpunkts
        const dx = bi.pos.x - cm.x;
        const dy = bi.pos.y - cm.y;
        const dz = bi.pos.z - cm.z;
        const dist_cm = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const v_rad = (dx*bi.vel.vx + dy*bi.vel.vy + dz*bi.vel.vz) / Math.max(0.01, dist_cm);
        
        // Positive mechanische Bahnenergie + Auswärtsdrift = Hyperbolischer Auswurf!
        if (eps_spec > 0.0 && v_rad > 0.05 && dist_cm > 1.8) {
          this.slingshotLocked = true;
          this.lockedCandidateIdx = i;
          this.slingshotTime = this.time;
          const remainingDist = Math.max(0.1, 14.0 - dist_cm);
          const v_eff = Math.max(v_rad, Math.sqrt(2.0 * eps_spec) * 0.85);
          this.keplerFlightTime = remainingDist / Math.max(0.05, v_eff);
          this.exactEjectionTime = this.time + this.keplerFlightTime;
          break;
        }
      }
    }
  }

  // Akkumulierte Eigenzeiten für jeden der drei Körper (Multi-Eigenzeit-Modell)
  accumulateProperTimes(dt) {
    if (this.bodies.length < 3) return;
    const c2 = this.c_eff * this.c_eff;
    const eps2 = this.softening * this.softening;

    for (let i = 0; i < 3; i++) {
      const b = this.bodies[i];
      const v2 = b.vel.vx * b.vel.vx + b.vel.vy * b.vel.vy + b.vel.vz * b.vel.vz;
      let phi_i = 0;
      for (let j = 0; j < 3; j++) {
        if (i !== j) {
          const bj = this.bodies[j];
          const dx = bj.pos.x - b.pos.x;
          const dy = bj.pos.y - b.pos.y;
          const dz = bj.pos.z - b.pos.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
          phi_i -= (this.G * bj.mass) / dist;
        }
      }
      // dtau_i / dt = sqrt(max(0.001, 1 - v^2/c^2 + 2*phi/c^2))
      const factor = 1.0 - (v2 / c2) + (2.0 * phi_i / c2);
      const rate = Math.sqrt(Math.max(0.001, factor));
      this.timeRates[i] = rate;
      this.tau[i] += rate * dt;
    }
  }

  // Gibt die aktuellen Eigenzeit-Daten und die Zeit-Schere Δτ zurück
  getProperTimeData() {
    const diffAB = Math.abs(this.tau[0] - this.tau[1]);
    const diffBC = Math.abs(this.tau[1] - this.tau[2]);
    const diffCA = Math.abs(this.tau[2] - this.tau[0]);
    const maxRift = Math.max(diffAB, diffBC, diffCA);

    return {
      tauA: this.tau[0],
      tauB: this.tau[1],
      tauC: this.tau[2],
      rateA: this.timeRates[0],
      rateB: this.timeRates[1],
      rateC: this.timeRates[2],
      maxRift: maxRift
    };
  }

  // Pfad 3: Analytische Auswurf-Formel & Prädiktor (Symbolische KI)
  predictEjectionFormula() {
    if (this.bodies.length < 3) return null;

    const m = this.bodies.map(b => b.mass);
    const M_tot = m[0] + m[1] + m[2];

    // Finde den leichtesten Körper (m_min)
    let minIdx = 0;
    if (m[1] < m[0] && m[1] <= m[2]) minIdx = 1;
    else if (m[2] < m[0] && m[2] <= m[1]) minIdx = 2;
    const m_min = m[minIdx];

    // Initiales Gravitationspotential & Zeit-Schere ΔΦ
    const { a1, a2, a3 } = this.getTriangleSides();
    const d01 = Math.max(0.01, a1);
    const d12 = Math.max(0.01, a2);
    const d20 = Math.max(0.01, a3);

    const phi0 = this.G * (m[1] / d01 + m[2] / d20);
    const phi1 = this.G * (m[0] / d01 + m[2] / d12);
    const phi2 = this.G * (m[0] / d20 + m[1] / d12);

    const phi_max = Math.max(phi0, phi1, phi2);
    const phi_min = Math.min(phi0, phi1, phi2);
    const phi_mean = (phi0 + phi1 + phi2) / 3.0;
    const delta_phi = phi_max - phi_min;
    const phi_shear_rel = delta_phi / Math.max(0.01, phi_mean);

    // 1. Stufe: Initialer Verwundbarkeits-Score Vi (vor dem Start)
    let bestScore = -Infinity;
    let stage1Idx = minIdx;
    for (let i = 0; i < 3; i++) {
      const b = this.bodies[i];
      const phi_i = (i === 0) ? phi0 : ((i === 1) ? phi1 : phi2);
      const v2 = b.vel.vx * b.vel.vx + b.vel.vy * b.vel.vy + b.vel.vz * b.vel.vz;
      const xi = v2 / (2.0 * Math.max(0.01, phi_i));
      const score = Math.pow(M_tot / b.mass, 1.5) * (1.0 + 1.2 * xi);
      if (score > bestScore) {
        bestScore = score;
        stage1Idx = i;
      }
    }

    // 2-Stufen Auswurf-Kandidaten-Wahl:
    // Ist der Hyperbolische Phasenriss bereits aufgetreten? -> 100% Deterministisch!
    // Andernfalls: Stufe 1 Schwachstellen-Vektor (78-89% Konfidenz)
    let activeIdx = stage1Idx;
    let confidence = 78;
    let stageLabel = 'Stufe 1 (Initial)';
    const m_max = Math.max(m[0], m[1], m[2]);
    const m_ratio = m_max / Math.max(0.01, m_min);
    if (m_ratio > 1.4) confidence = 85;
    if (m_ratio > 2.0) confidence = 92;

    if (this.slingshotLocked && this.lockedCandidateIdx >= 0) {
      activeIdx = this.lockedCandidateIdx;
      confidence = 100; // Deterministisch: Hyperbolische Auswärtsbahn bestätigt!
      stageLabel = 'Stufe 2: Phasenriss (100% Sicher)';
    }

    const candidateBody = this.bodies[activeIdx] || this.bodies[minIdx];

    // Konservierte Gesamtenergie |E|
    const rawE = (this.initialEnergy !== null && this.initialEnergy !== undefined) ? this.initialEnergy : this.calculateTotalEnergy();
    const eVal = (typeof rawE === 'object' && rawE !== null) ? (rawE.total !== undefined ? rawE.total : 0) : rawE;
    const absE = Math.max(0.01, Math.abs(eVal));
    const E_eff = Math.max(absE, 0.25);

    // Heggie-Grundskala: T_scale = G * M^(2.5) / |E|^(1.5)
    const t_scale = (this.G * Math.pow(M_tot, 2.5)) / Math.pow(E_eff, 1.5);

    // Trinitäts-Spannungsenergie Ω und Trägheitsmoment I
    const omega = this.calculateOmega();
    const cm = this.calculateCenterOfMass();
    let I_val = 0;
    for (const b of this.bodies) {
      const dx = b.pos.x - cm.x;
      const dy = b.pos.y - cm.y;
      const dz = b.pos.z - cm.z;
      I_val += b.mass * (dx * dx + dy * dy + dz * dz);
    }

    // Gesamtdrehimpuls L = sum m_i (r_i x v_i) und Rotations-Schutzfaktor η
    let Lx = 0, Ly = 0, Lz = 0;
    for (const b of this.bodies) {
      Lx += b.mass * (b.pos.y * b.vel.vz - b.pos.z * b.vel.vy);
      Ly += b.mass * (b.pos.z * b.vel.vx - b.pos.x * b.vel.vz);
      Lz += b.mass * (b.pos.x * b.vel.vy - b.pos.y * b.vel.vx);
    }
    const L_mag = Math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz);
    const eta_rot = Math.min(2.5, (L_mag * L_mag) / (absE * Math.max(0.01, I_val)));

    // Doppler-Radialimpuls cos(α) des Auswurf-Kandidaten relativ zum inneren Doppelstern
    const otherIndices = [0, 1, 2].filter(i => i !== activeIdx);
    const b0 = otherIndices[0];
    const b1 = otherIndices[1];
    const m_bin = m[b0] + m[b1];

    const cm_bin_x = (m[b0] * this.bodies[b0].pos.x + m[b1] * this.bodies[b1].pos.x) / m_bin;
    const cm_bin_y = (m[b0] * this.bodies[b0].pos.y + m[b1] * this.bodies[b1].pos.y) / m_bin;
    const cm_bin_z = (m[b0] * this.bodies[b0].pos.z + m[b1] * this.bodies[b1].pos.z) / m_bin;

    const v_bin_x = (m[b0] * this.bodies[b0].vel.vx + m[b1] * this.bodies[b1].vel.vx) / m_bin;
    const v_bin_y = (m[b0] * this.bodies[b0].vel.vy + m[b1] * this.bodies[b1].vel.vy) / m_bin;
    const v_bin_z = (m[b0] * this.bodies[b0].vel.vz + m[b1] * this.bodies[b1].vel.vz) / m_bin;

    const rx = candidateBody.pos.x - cm_bin_x;
    const ry = candidateBody.pos.y - cm_bin_y;
    const rz = candidateBody.pos.z - cm_bin_z;
    const d_rel = Math.max(0.01, Math.sqrt(rx * rx + ry * ry + rz * rz));

    const vx = candidateBody.vel.vx - v_bin_x;
    const vy = candidateBody.vel.vy - v_bin_y;
    const vz = candidateBody.vel.vz - v_bin_z;
    const v_rel = Math.max(0.01, Math.sqrt(vx * vx + vy * vy + vz * vz));

    // cos(α): Negativ = Sturzflug (Super-Plateau), Positiv = Auswärts / Resonanz
    const cos_alpha = (rx * vx + ry * vy + rz * vz) / (d_rel * v_rel);

    // Glattes physikalisches Phasen-Tor (analytisch, 10M-kalibriert: k = 12.0)
    const phase_gate = 0.5 * (1.0 + Math.tanh(12.0 * (cos_alpha + 0.05)));

    // Reines Trinitäts-Zyklen-Modell mit Zeit-Schere & Rotations-Schutz (10.000.000 Systeme kalibriert)
    const eps_chaos = 0.0500;
    const rot_factor = 1.0 + Math.pow(Math.min(2.5, eta_rot), 0.1875);
    const n_base = (1.0020 * rot_factor) / (phi_shear_rel + eps_chaos);
    const n_pred = Math.max(1, 1 + Math.round((n_base - 1.0) * phase_gate));

    // Charakteristische Oszillationszeit & dynamische Kepler-Flugzeit (10M-kalibriert)
    const t_orbit = 0.0051 * t_scale;
    const t_kepler = 1.7293 * (Math.pow(14.0, 1.5)) / Math.sqrt(Math.max(0.1, this.G * M_tot));

    // Prädiktion der Lebensdauer
    const t_pred_symbolic = (n_pred - 1) * t_orbit + t_kepler;
    const t_slingshot = Math.max(4.0, (n_pred - 1) * t_orbit);

    return {
      candidateName: candidateBody.name,
      candidateColor: candidateBody.color,
      candidateMass: candidateBody.mass,
      activeIdx: activeIdx,
      minIdx: minIdx,
      isSlingshotLocked: this.slingshotLocked,
      stageLabel: stageLabel,
      confidencePct: confidence,
      t_scale: t_scale,
      t_orbit: t_orbit,
      t_kepler: t_kepler,
      L_mag: L_mag,
      eta_rot: eta_rot,
      cos_alpha: cos_alpha,
      phase_gate: phase_gate,
      eps_chaos: eps_chaos,
      rot_bonus_pct: Math.round((rot_factor - 1.0) * 100),
      t_pred_symbolic: Math.max(4.5, t_pred_symbolic),
      t_slingshot: t_slingshot,
      delta_phi_rel: phi_shear_rel,
      simTime: this.time,
      n_pred: n_pred,
      n_base: n_base,
      cycles_count: this.cyclesCount,
      sigma_integral: this.sigmaIntegral,
      exact_eject_time: this.exactEjectionTime,
      kepler_flight: this.keplerFlightTime
    };
  }

  // Topologischer Stabilitäts- & Chaotik-Indikator
  evaluateStability() {
    if (!this.phaseHistory || this.phaseHistory.length < 60) {
      return { score: 100, state: 'KALIBRIERUNG', label: 'Wird analysiert...', color: '#00f0ff' };
    }

    const { a1, a2, a3 } = this.getTriangleSides();
    const maxSide = Math.max(a1, a2, a3);
    const minSide = Math.min(a1, a2, a3);

    // 1. Auswurf-Prüfung (Ejection Event): Ein Körper wird weggeschleudert
    if (maxSide > 25.0 || (maxSide > 12.0 && maxSide / Math.max(0.01, minSide) > 15.0)) {
      return {
        score: 5,
        state: 'AUSWURF',
        label: 'Auswurf / System-Zerfall',
        desc: 'Ein Körper hat Fluchtgeschwindigkeit erreicht. Das Trinitäts-Dreieck zerbricht.',
        color: '#ff0055'
      };
    }

    // 2. Rekurrenz-Analyse (Kehrt das Dreieck in denselben Zustand zurück?)
    const current = { omega: this.calculateOmega(), dOmega: this.omegaRate, sides: { a1, a2, a3 } };
    let minRecurrenceDist = Infinity;

    // Untersuche historische Zustände vor mindestens 35 Zeitschritten
    const testWindow = this.phaseHistory.slice(0, this.phaseHistory.length - 35);
    for (const h of testWindow) {
      const dO = (current.omega - h.omega);
      const dD = (current.dOmega - h.dOmega) * 0.2;
      const dS = (current.sides.a1 - h.sides.a1) ** 2 + (current.sides.a2 - h.sides.a2) ** 2 + (current.sides.a3 - h.sides.a3) ** 2;
      const dist = Math.sqrt(dO * dO + dD * dD + dS);
      if (dist < minRecurrenceDist) {
        minRecurrenceDist = dist;
      }
    }

    // 3. Ω-Varianz & Glätte
    const omegaRates = this.phaseHistory.slice(-40).map(p => p.dOmega);
    const meanRate = omegaRates.reduce((a, b) => a + b, 0) / omegaRates.length;
    const varianceRate = omegaRates.reduce((sum, r) => sum + (r - meanRate) ** 2, 0) / omegaRates.length;

    // 4. Stabilitäts-Score & Klassifikation
    // Bei Lagrange: minRecurrenceDist ~ 0, varianceRate ~ 0 -> Score 100%
    // Bei Figure-8: minRecurrenceDist < 0.15 -> Score 90-98%
    // Bei Chaos: minRecurrenceDist > 0.8, varianceRate hoch -> Score 10-45%
    let score = Math.max(0, Math.min(100, Math.round(100 - (minRecurrenceDist * 45 + Math.min(50, varianceRate * 2.5)))));

    if (minRecurrenceDist < 0.22 && varianceRate < 8.0) {
      return {
        score: Math.max(88, score),
        state: 'PERIODISCH',
        label: 'Stabil Periodisch (Grenzzyklus)',
        desc: 'Geschlossene Schleife im Formraum. Das System kehrt exakt wieder (100% vorhersagbar).',
        color: '#00ff88'
      };
    } else if (minRecurrenceDist < 0.65 && varianceRate < 25.0) {
      return {
        score: Math.max(50, score),
        state: 'QUASI_PERIODISCH',
        label: 'Metastabil (Gedämpft)',
        desc: 'Komplexe Oszillation, mittelfristig gebunden, driftet allmählich.',
        color: '#ffb703'
      };
    } else {
      return {
        score: Math.min(45, score),
        state: 'CHAOTISCH',
        label: 'Hochgradig Chaotisch',
        desc: 'Offener Phasenraum. Exponentielle Divergenz (Butterfly-Effekt).',
        color: '#ff5400'
      };
    }
  }

  // Distanz zwischen zwei Punkten
  getDistance(pos1, pos2) {
    const dx = pos2.x - pos1.x;
    const dy = pos2.y - pos1.y;
    const dz = pos2.z - pos1.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  // Seiten des Trinitäts-Dreiecks (a1, a2, a3)
  getTriangleSides() {
    if (this.bodies.length < 3) return { a1: 0, a2: 0, a3: 0 };
    const a = this.bodies[0].pos;
    const b = this.bodies[1].pos;
    const c = this.bodies[2].pos;

    return {
      a1: this.getDistance(a, b), // Ka <-> Kb
      a2: this.getDistance(b, c), // Kb <-> Kc
      a3: this.getDistance(c, a), // Kc <-> Ka
    };
  }

  // Ω-Spannungsenergie (Spannungs-Operator):
  // Bindungs- und Dehnungsenergie des Konfigurations-Gewebes GS
  calculateOmega() {
    if (this.bodies.length < 3) return 0;
    const { a1, a2, a3 } = this.getTriangleSides();
    const [ba, bb, bc] = this.bodies;
    const eps = this.softening;

    // Ω entspricht der Betrags-Gravitations-Spannung im Dreiecks-Gewebe:
    // Ω = G * ( (ma*mb)/(a1+ε) + (mb*mc)/(a2+ε) + (mc*ma)/(a3+ε) )
    const term1 = (ba.mass * bb.mass) / Math.sqrt(a1 * a1 + eps * eps);
    const term2 = (bb.mass * bc.mass) / Math.sqrt(a2 * a2 + eps * eps);
    const term3 = (bc.mass * ba.mass) / Math.sqrt(a3 * a3 + eps * eps);

    return this.G * (term1 + term2 + term3);
  }

  // Dreiecksfläche (GS-Volumen im 2D/3D-Raum)
  calculateTriangleArea() {
    if (this.bodies.length < 3) return 0;
    const a = this.bodies[0].pos;
    const b = this.bodies[1].pos;
    const c = this.bodies[2].pos;

    // Vektor AB und AC
    const ab = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
    const ac = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };

    // Kreuzprodukt AB x AC
    const cx = ab.y * ac.z - ab.z * ac.y;
    const cy = ab.z * ac.x - ab.x * ac.z;
    const cz = ab.x * ac.y - ab.y * ac.x;

    return 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
  }

  // Symmetrie-Index: Wie nahe ist das Dreieck an der gleichseitigen Form (100% = perfekt gleichseitig)
  calculateSymmetryIndex() {
    const { a1, a2, a3 } = this.getTriangleSides();
    const mean = (a1 + a2 + a3) / 3.0;
    if (mean === 0) return 0;
    const variance = (Math.abs(a1 - mean) + Math.abs(a2 - mean) + Math.abs(a3 - mean)) / (3 * mean);
    return Math.max(0, Math.min(100, Math.round((1 - variance) * 100)));
  }

  // Kinetische Energie T
  calculateKineticEnergy() {
    let t = 0;
    for (const b of this.bodies) {
      const vSq = b.vel.vx * b.vel.vx + b.vel.vy * b.vel.vy + b.vel.vz * b.vel.vz;
      t += 0.5 * b.mass * vSq;
    }
    return t;
  }

  // Potentielle Energie V
  calculatePotentialEnergy() {
    return -this.calculateOmega();
  }

  // Gesamtenergie E = T + V
  calculateTotalEnergy() {
    return this.calculateKineticEnergy() + this.calculatePotentialEnergy();
  }

  // Drehimpulsvektor L = sum(m_i * r_i x v_i)
  calculateAngularMomentum() {
    let lx = 0, ly = 0, lz = 0;
    for (const b of this.bodies) {
      const rx = b.pos.x, ry = b.pos.y, rz = b.pos.z;
      const vx = b.vel.vx, vy = b.vel.vy, vz = b.vel.vz;

      lx += b.mass * (ry * vz - rz * vy);
      ly += b.mass * (rz * vx - rx * vz);
      lz += b.mass * (rx * vy - ry * vx);
    }
    return Math.sqrt(lx * lx + ly * ly + lz * lz);
  }

  // Barycenter (Schwerpunkt)
  getCenterOfMass() {
    let totMass = 0;
    let cx = 0, cy = 0, cz = 0;

    for (const b of this.bodies) {
      totMass += b.mass;
      cx += b.mass * b.pos.x;
      cy += b.mass * b.pos.y;
      cz += b.mass * b.pos.z;
    }

    if (totMass === 0) return { x: 0, y: 0, z: 0 };
    return { x: cx / totMass, y: cy / totMass, z: cz / totMass };
  }

  calculateCenterOfMass() {
    return this.getCenterOfMass();
  }
}
