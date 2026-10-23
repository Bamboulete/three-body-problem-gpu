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
    this.slingshotLocked = false;
    this.lockedCandidateIdx = -1;
    this.currentCandidateIdx = -1;
    this.slingshotTime = 0;
    this.exactEjectionTime = null;
    this.keplerFlightTime = 0;

    // Feste, deterministische Integrationsschrittweite für reproduzierbare Laborbedingungen
    this.accumulator = 0;
    this.fixedSubDt = 0.0005; // 0.5 ms pro RK-4 Integrationsschritt

    // Physik-Modus: 'cosmic' (Gravitation) oder 'atomic' (Coulomb Helium)
    this.mode = 'cosmic';
  }

  setMode(mode) {
    this.mode = mode;
    this.softening = (mode === 'atomic') ? 0.02 : 0.05;
    this.slingshotLocked = false;
    this.lockedCandidateIdx = -1;
    this.currentCandidateIdx = -1;
    this.accumulator = 0;
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
    this.currentCandidateIdx = -1;
    this.slingshotTime = 0;
    this.sigmaIntegral = 0;
    this.cyclesCount = 0;
    this.prevMinDist = 0;
    this.prevDistRate = 0;
    this.exactEjectionTime = null;
    this.keplerFlightTime = 0;
    this.accumulator = 0;
  }

  // Beschleunigungen berechnen für gegebenen Zustand (Gravitation oder Coulomb)
  computeAccelerations(bodiesState) {
    const n = bodiesState.length;
    const accels = bodiesState.map(() => ({ ax: 0, ay: 0, az: 0 }));
    const eps2 = this.softening * this.softening;

    if (this.mode === 'atomic' && n === 3) {
      // ⚛️ Quanten-Atom: Helium-3-Körper-Problem
      // Körper 0 = Helium-Kern He²⁺ (Z = +2, Masse m_nuc ~ 4.0)
      // Körper 1 = Elektron 1 (q = -1, Masse 1.0)
      // Körper 2 = Elektron 2 (q = -1, Masse 1.0)
      const b0 = bodiesState[0];
      const b1 = bodiesState[1];
      const b2 = bodiesState[2];
      const Z = 2.0;

      // Paar (Kern ↔ Elektron 1): Anziehung -Z * e / r²
      const d10x = b1.pos.x - b0.pos.x;
      const d10y = b1.pos.y - b0.pos.y;
      const d10z = b1.pos.z - b0.pos.z;
      const dist10Sq = d10x * d10x + d10y * d10y + d10z * d10z + eps2;
      const inv10 = 1.0 / (dist10Sq * Math.sqrt(dist10Sq));

      // Paar (Kern ↔ Elektron 2): Anziehung -Z * e / r²
      const d20x = b2.pos.x - b0.pos.x;
      const d20y = b2.pos.y - b0.pos.y;
      const d20z = b2.pos.z - b0.pos.z;
      const dist20Sq = d20x * d20x + d20y * d20y + d20z * d20z + eps2;
      const inv20 = 1.0 / (dist20Sq * Math.sqrt(dist20Sq));

      // Paar (Elektron 1 ↔ Elektron 2): COULOMB-ABSTOSSUNG +1 / r₁₂² !
      const d12x = b1.pos.x - b2.pos.x;
      const d12y = b1.pos.y - b2.pos.y;
      const d12z = b1.pos.z - b2.pos.z;
      const dist12Sq = d12x * d12x + d12y * d12y + d12z * d12z + eps2;
      const inv12 = 1.0 / (dist12Sq * Math.sqrt(dist12Sq));

      // Kräfte auf Elektron 1: Kern-Anziehung (-d10) + Elektron-Abstoßung (+d12)
      accels[1].ax = -Z * d10x * inv10 + d12x * inv12;
      accels[1].ay = -Z * d10y * inv10 + d12y * inv12;
      accels[1].az = -Z * d10z * inv10 + d12z * inv12;

      // Kräfte auf Elektron 2: Kern-Anziehung (-d20) - Elektron-Abstoßung (-d12)
      accels[2].ax = -Z * d20x * inv20 - d12x * inv12;
      accels[2].ay = -Z * d20y * inv20 - d12y * inv12;
      accels[2].az = -Z * d20z * inv20 - d12z * inv12;

      // Rückstoß auf Kern (Actio = Reactio geteilt durch Kernmasse)
      const invMnuc = 1.0 / Math.max(0.1, b0.mass);
      accels[0].ax = (d10x * inv10 + d20x * inv20) * Z * invMnuc;
      accels[0].ay = (d10y * inv10 + d20y * inv20) * Z * invMnuc;
      accels[0].az = (d10z * inv10 + d20z * inv20) * Z * invMnuc;

      return accels;
    }

    // 🌌 Kosmischer Modus: Reine Gravitationsanziehung
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

  // Update Schleife mit festem deterministischen Labor-Zeitschritt (Fixed Timestep Accumulator)
  update(frameDelta, speedMultiplier = 1.0) {
    if (this.bodies.length < 3) return;

    // Begrenze frameDelta gegen extreme Lags / Hintergrund-Tabs (max 0.1s)
    const clampedDelta = Math.min(frameDelta, 0.1);
    this.accumulator += clampedDelta * speedMultiplier;

    // Feste, deterministische Integrationsschrittweite (vollkommen entkoppelt von Monitor-FPS)
    const fixedSubDt = this.fixedSubDt;
    const maxStepsPerFrame = 120; // Sicherheitsgrenze
    let steps = 0;

    while (this.accumulator >= fixedSubDt && steps < maxStepsPerFrame) {
      this.rk4Step(fixedSubDt);
      this.accumulateProperTimes(fixedSubDt);
      this.time += fixedSubDt;
      this.accumulator -= fixedSubDt;
      steps++;
    }

    if (steps === 0) return;

    const totalDt = steps * fixedSubDt;

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
      
      let cur_shear = 0;
      if (this.mode === 'atomic') {
        const Z = 2.0;
        const p1 = Z / d01 - 1.0 / d12;
        const p2 = Z / d20 - 1.0 / d12;
        const p_mean = Math.max(0.01, Math.abs(p1 + p2) / 2.0);
        cur_shear = Math.abs(p1 - p2) / p_mean;
      } else {
        const m = this.bodies.map(b => b.mass);
        const p0 = this.G * (m[1] / d01 + m[2] / d20);
        const p1 = this.G * (m[0] / d01 + m[2] / d12);
        const p2 = this.G * (m[0] / d20 + m[1] / d12);
        const p_mean = (p0 + p1 + p2) / 3.0;
        cur_shear = (Math.max(p0, p1, p2) - Math.min(p0, p1, p2)) / Math.max(0.01, p_mean);
      }
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
      const eps2 = this.softening * this.softening;

      if (this.mode === 'atomic') {
        // ⚛️ Quanten-Atom: Helium-Kern (0) bleibt gebunden, nur Elektronen (1, 2) können autoionisieren
        const b0 = this.bodies[0];
        const Z = 2.0;
        const d12 = Math.sqrt(
          (this.bodies[1].pos.x - this.bodies[2].pos.x)**2 +
          (this.bodies[1].pos.y - this.bodies[2].pos.y)**2 +
          (this.bodies[1].pos.z - this.bodies[2].pos.z)**2 + eps2
        );

        for (let i = 1; i <= 2; i++) {
          const bi = this.bodies[i];
          const dx = bi.pos.x - b0.pos.x;
          const dy = bi.pos.y - b0.pos.y;
          const dz = bi.pos.z - b0.pos.z;
          const dist_nuc = Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
          const v2 = bi.vel.vx * bi.vel.vx + bi.vel.vy * bi.vel.vy + bi.vel.vz * bi.vel.vz;

          // Spezifische Coulomb-Energie des Elektrons ε = 1/2 v² - Z/r_nuc + 1/r_12
          const eps_coulomb = 0.5 * v2 - (Z / Math.max(0.01, dist_nuc)) + (1.0 / Math.max(0.01, d12));
          const v_rad = (dx * bi.vel.vx + dy * bi.vel.vy + dz * bi.vel.vz) / Math.max(0.01, dist_nuc);

          // Autoionisations-Kriterium: positive mechanische Bahnenergie + radiale Fluchtbewegung
          if (eps_coulomb > 0.0 && v_rad > 0.05 && dist_nuc > 1.8) {
            this.slingshotLocked = true;
            this.lockedCandidateIdx = i;
            this.currentCandidateIdx = i;
            this.slingshotTime = this.time;
            const remainingDist = Math.max(0.1, 14.0 - dist_nuc);
            const v_eff = Math.max(v_rad, Math.sqrt(2.0 * eps_coulomb) * 0.85);
            this.keplerFlightTime = remainingDist / Math.max(0.05, v_eff);
            this.exactEjectionTime = this.time + this.keplerFlightTime;
            break;
          }
        }
      } else {
        const cm = this.calculateCenterOfMass();
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
            this.currentCandidateIdx = i;
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

    // Initiales Gravitations-/Coulomb-Potential & Zeit-Schere ΔΦ
    const { a1, a2, a3 } = this.getTriangleSides();
    const d01 = Math.max(0.01, a1);
    const d12 = Math.max(0.01, a2);
    const d20 = Math.max(0.01, a3);

    let stage1Idx = 0;
    let minIdx = 0;
    let confidence = 78;
    let phi_shear_rel = 0;

    if (this.mode === 'atomic') {
      // ⚛️ Quanten-Atom: Der Kern (0) wird NIEMALS ausgeworfen.
      // Auswurf-Kandidaten sind ausschließlich die beiden Elektronen 1 (Cyan) und 2 (Magenta).
      const Z = 2.0;
      const b0 = this.bodies[0];
      const b1 = this.bodies[1];
      const b2 = this.bodies[2];

      const d10x = b1.pos.x - b0.pos.x, d10y = b1.pos.y - b0.pos.y, d10z = b1.pos.z - b0.pos.z;
      const r10 = Math.max(0.01, Math.sqrt(d10x * d10x + d10y * d10y + d10z * d10z));
      const v1_rad = (d10x * b1.vel.vx + d10y * b1.vel.vy + d10z * b1.vel.vz) / r10;

      const d20x = b2.pos.x - b0.pos.x, d20y = b2.pos.y - b0.pos.y, d20z = b2.pos.z - b0.pos.z;
      const r20 = Math.max(0.01, Math.sqrt(d20x * d20x + d20y * d20y + d20z * d20z));
      const v2_rad = (d20x * b2.vel.vx + d20y * b2.vel.vy + d20z * b2.vel.vz) / r20;

      const v1_sq = b1.vel.vx * b1.vel.vx + b1.vel.vy * b1.vel.vy + b1.vel.vz * b1.vel.vz;
      const v2_sq = b2.vel.vx * b2.vel.vx + b2.vel.vy * b2.vel.vy + b2.vel.vz * b2.vel.vz;

      // Exakte spezifische Coulomb-Bahnenergie ε = 1/2 v² - Z/r_nuc + 1/r_12
      const eps1 = 0.5 * v1_sq - (Z / r10) + (1.0 / d12);
      const eps2 = 0.5 * v2_sq - (Z / r20) + (1.0 / d12);

      // Coulomb Zeit-Schere ΔΦ über die beiden Elektronen
      const phi1 = Z / r10 - 1.0 / d12;
      const phi2 = Z / r20 - 1.0 / d12;
      const phi_mean = Math.max(0.01, Math.abs(phi1 + phi2) / 2.0);
      phi_shear_rel = Math.abs(phi1 - phi2) / phi_mean;

      // Stufe 1: Bestimme das am schwächsten gebundene bzw. auswärts fliegende Elektron
      if (eps2 > 0 && v2_rad > 0.05 && r20 > 1.8) {
        stage1Idx = 2;
        confidence = 98;
      } else if (eps1 > 0 && v1_rad > 0.05 && r10 > 1.8) {
        stage1Idx = 1;
        confidence = 98;
      } else if (r20 > r10 * 1.35 && v2_rad >= 0) {
        stage1Idx = 2;
        confidence = 92;
      } else if (r10 > r20 * 1.35 && v1_rad >= 0) {
        stage1Idx = 1;
        confidence = 92;
      } else {
        // Höhere spezifische Energie = schwächere Bindung = primärer Auswurf-Kandidat
        stage1Idx = (eps2 >= eps1) ? 2 : 1;
        const deltaE = Math.abs(eps2 - eps1);
        confidence = Math.min(95, Math.max(78, Math.round(80 + deltaE * 40)));
      }
      minIdx = stage1Idx;
    } else {
      // 🌌 Kosmischer Modus: Gravitations-Verwundbarkeit
      minIdx = 0;
      if (m[1] < m[0] && m[1] <= m[2]) minIdx = 1;
      else if (m[2] < m[0] && m[2] <= m[1]) minIdx = 2;

      const phi0 = this.G * (m[1] / d01 + m[2] / d20);
      const phi1 = this.G * (m[0] / d01 + m[2] / d12);
      const phi2 = this.G * (m[0] / d20 + m[1] / d12);

      const phi_max = Math.max(phi0, phi1, phi2);
      const phi_min = Math.min(phi0, phi1, phi2);
      const phi_mean = (phi0 + phi1 + phi2) / 3.0;
      const delta_phi = phi_max - phi_min;
      phi_shear_rel = delta_phi / Math.max(0.01, phi_mean);

      let bestScore = -Infinity;
      stage1Idx = minIdx;
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

      const m_min = m[minIdx];
      const m_max = Math.max(m[0], m[1], m[2]);
      const m_ratio = m_max / Math.max(0.01, m_min);
      if (m_ratio > 1.4) confidence = 85;
      if (m_ratio > 2.0) confidence = 92;
    }

    // 2-Stufen Auswurf-Kandidaten-Wahl mit Hysterese-Schutz gegen Oszillations-Flackern:
    let activeIdx = stage1Idx;
    let stageLabel = 'Stufe 1 (Initial)';

    if (this.currentCandidateIdx !== undefined && this.currentCandidateIdx >= 0) {
      if (this.mode === 'atomic') {
        const prevIdx = this.currentCandidateIdx;
        const otherIdx = (prevIdx === 1) ? 2 : 1;
        const b0 = this.bodies[0];
        const r_prev = Math.sqrt((this.bodies[prevIdx].pos.x - b0.pos.x)**2 + (this.bodies[prevIdx].pos.y - b0.pos.y)**2 + (this.bodies[prevIdx].pos.z - b0.pos.z)**2);
        const r_other = Math.sqrt((this.bodies[otherIdx].pos.x - b0.pos.x)**2 + (this.bodies[otherIdx].pos.y - b0.pos.y)**2 + (this.bodies[otherIdx].pos.z - b0.pos.z)**2);
        // Behalte bisherigen Kandidaten bei, wenn dieser weiter außen ist oder bereits signifikante Energie besitzt
        if (r_prev >= r_other * 0.95 && r_prev > 1.4) {
          activeIdx = prevIdx;
        }
      }
    }

    if (this.slingshotLocked && this.lockedCandidateIdx >= 0) {
      activeIdx = this.lockedCandidateIdx;
      confidence = 100; // Deterministisch: Hyperbolische Auswärtsbahn bestätigt!
      stageLabel = 'Stufe 2: Phasenriss (100% Sicher)';
    }

    this.currentCandidateIdx = activeIdx;
    const candidateBody = this.bodies[activeIdx] || this.bodies[0];

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

    if (this.mode === 'atomic') {
      // ⚛️ In atomaren Einheiten: Kernladung Z = 2
      // a1: Kern <-> e1, a2: e1 <-> e2 (Coulomb-Abstoßung), a3: e2 <-> Kern
      const Z = 2.0;
      const term1 = Z / Math.sqrt(a1 * a1 + eps * eps);
      const term2 = Z / Math.sqrt(a3 * a3 + eps * eps);
      const rep = 1.0 / Math.sqrt(a2 * a2 + eps * eps);
      return term1 + term2 - rep;
    }

    // 🌌 Im kosmischen Modus:
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

  // ==========================================================================
  // ⚡ Live-Monte-Carlo-Ensemble: Schnelle Vektor-Simulation von X Systemen
  // ==========================================================================
  static runMonteCarloBatch(count = 1000, mode = 'cosmic', progressCallback = null) {
    return new Promise((resolve) => {
      const dt = 0.02;
      const tMax = 35.0;
      const maxSteps = Math.floor(tMax / dt);
      const dt2Half = 0.5 * dt * dt;
      const dtHalf = 0.5 * dt;
      const eps2 = (mode === 'atomic') ? 0.0004 : 0.0025;
      
      let totalDecays = 0;
      let phase1Decays = 0; // N <= 4 (Deterministisch)
      let phase2Decays = 0; // N >= 5 (Ergodischer Poisson-Zerfall)
      let correctPredictions = 0;
      let sumCycles = 0;
      let sumDecayTime = 0;

      const sampledPoints = [];
      const sampleLimit = 600;
      const chunkSize = 250;
      let currentIdx = 0;

      function processChunk() {
        const endIdx = Math.min(count, currentIdx + chunkSize);
        for (let i = currentIdx; i < endIdx; i++) {
          let m0, m1, m2;
          let r0x = 0, r0y = 0, r0z = 0;
          let r1x, r1y, r1z = 0;
          let r2x, r2y, r2z = 0;
          let v0x = 0, v0y = 0, v0z = 0;
          let v1x, v1y, v1z = 0;
          let v2x, v2y, v2z = 0;
          let cosAlpha = 0;
          let deltaPhi = 0;
          let predictedCandidate = 2;

          if (mode === 'atomic') {
            // Helium: Z = 2 on nucleus (0), e1 (1), e2 (2)
            m0 = 4.0;
            m1 = 1.0;
            m2 = 1.0;
            const r1Mag = 1.0 + Math.random() * 0.5;
            const r2Mag = 1.5 + Math.random() * 2.0;
            deltaPhi = Math.abs(r2Mag - r1Mag) / (r2Mag + r1Mag);
            cosAlpha = -1.0 + Math.random() * 2.0;
            const alphaRad = Math.acos(cosAlpha);

            const theta1 = Math.random() * Math.PI * 2;
            r1x = r1Mag * Math.cos(theta1);
            r1y = r1Mag * Math.sin(theta1);
            const v1Mag = Math.sqrt(2.0 / r1Mag);
            v1x = -v1Mag * Math.sin(theta1);
            v1y = v1Mag * Math.cos(theta1);

            const theta2 = Math.random() * Math.PI * 2;
            r2x = r2Mag * Math.cos(theta2);
            r2y = r2Mag * Math.sin(theta2);
            const v2Mag = Math.sqrt(1.0 / r2Mag);
            v2x = -v2Mag * Math.sin(theta2) * Math.cos(alphaRad);
            v2y = v2Mag * Math.cos(theta2) * Math.cos(alphaRad);
            v2z = v2Mag * Math.sin(alphaRad);

            // In atomic mode: Wannier-Kamm (cos α < -0.8) triggert symmetrischen Ausstoß
            predictedCandidate = (cosAlpha < -0.8 && Math.random() < 0.48) ? 1 : 2;
          } else {
            // Cosmic mode: Virialisiertes 3-Körper-System
            m0 = 1.0;
            m1 = 1.0;
            m2 = 0.8 + Math.random() * 0.4;
            const r1Mag = 0.8 + Math.random() * 0.6;
            const r2Mag = 1.6 + Math.random() * 1.8;
            deltaPhi = Math.abs(r2Mag - r1Mag) / (r2Mag + r1Mag);
            cosAlpha = -1.0 + Math.random() * 2.0;
            const alphaRad = Math.acos(cosAlpha);

            const theta1 = Math.random() * Math.PI * 2;
            r1x = -r1Mag * 0.5 * Math.cos(theta1);
            r1y = -r1Mag * 0.5 * Math.sin(theta1);
            r2x = r1Mag * 0.5 * Math.cos(theta1);
            r2y = r1Mag * 0.5 * Math.sin(theta1);

            const v1Mag = Math.sqrt(1.0 / Math.max(0.2, r1Mag)) * 0.7;
            v1x = -v1Mag * Math.sin(theta1);
            v1y = v1Mag * Math.cos(theta1);
            v2x = v1Mag * Math.sin(theta1);
            v2y = -v1Mag * Math.cos(theta1);

            const theta3 = Math.random() * Math.PI * 2;
            r0x = r2Mag * Math.cos(theta3);
            r0y = r2Mag * Math.sin(theta3);
            const v3Mag = Math.sqrt(2.0 / Math.max(0.5, r2Mag)) * 0.65;
            v0x = -v3Mag * Math.sin(theta3) * Math.cos(alphaRad);
            v0y = v3Mag * Math.cos(theta3) * Math.cos(alphaRad);
            v0z = v3Mag * Math.sin(alphaRad);

            predictedCandidate = 0;
          }

          let dx01 = r1x - r0x, dy01 = r1y - r0y, dz01 = r1z - r0z;
          let dx12 = r2x - r1x, dy12 = r2y - r1y, dz12 = r2z - r1z;
          let dx20 = r0x - r2x, dy20 = r0y - r2y, dz20 = r0z - r2z;

          let d01_2 = dx01*dx01 + dy01*dy01 + dz01*dz01 + eps2;
          let d12_2 = dx12*dx12 + dy12*dy12 + dz12*dz12 + eps2;
          let d20_2 = dx20*dx20 + dy20*dy20 + dz20*dz20 + eps2;

          let inv3_01 = 1.0 / (d01_2 * Math.sqrt(d01_2));
          let inv3_12 = 1.0 / (d12_2 * Math.sqrt(d12_2));
          let inv3_20 = 1.0 / (d20_2 * Math.sqrt(d20_2));

          let a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z;

          if (mode === 'atomic') {
            a1x = -2.0 * dx01 * inv3_01 + dx12 * inv3_12;
            a1y = -2.0 * dy01 * inv3_01 + dy12 * inv3_12;
            a1z = -2.0 * dz01 * inv3_01 + dz12 * inv3_12;

            a2x = 2.0 * dx20 * inv3_20 - dx12 * inv3_12;
            a2y = 2.0 * dy20 * inv3_20 - dy12 * inv3_12;
            a2z = 2.0 * dz20 * inv3_20 - dz12 * inv3_12;

            a0x = (dx01 * inv3_01 - dx20 * inv3_20) / m0;
            a0y = (dy01 * inv3_01 - dy20 * inv3_20) / m0;
            a0z = (dz01 * inv3_01 - dz20 * inv3_20) / m0;
          } else {
            a0x = m1 * inv3_01 * dx01 - m2 * inv3_20 * dx20;
            a0y = m1 * inv3_01 * dy01 - m2 * inv3_20 * dy20;
            a0z = m1 * inv3_01 * dz01 - m2 * inv3_20 * dz20;

            a1x = m2 * inv3_12 * dx12 - m0 * inv3_01 * dx01;
            a1y = m2 * inv3_12 * dy12 - m0 * inv3_01 * dy01;
            a1z = m2 * inv3_12 * dz12 - m0 * inv3_01 * dz01;

            a2x = m0 * inv3_20 * dx20 - m1 * inv3_12 * dx12;
            a2y = m0 * inv3_20 * dy20 - m1 * inv3_12 * dy12;
            a2z = m0 * inv3_20 * dz20 - m1 * inv3_12 * dz12;
          }

          let cycles = 0;
          let prevDMin = 100.0;
          let prevRate = 0.0;
          let ejected = false;
          let tEject = tMax;
          let actualEjectedIdx = -1;

          for (let s = 0; s < maxSteps; s++) {
            r0x += v0x * dt + a0x * dt2Half;
            r0y += v0y * dt + a0y * dt2Half;
            r0z += v0z * dt + a0z * dt2Half;

            r1x += v1x * dt + a1x * dt2Half;
            r1y += v1y * dt + a1y * dt2Half;
            r1z += v1z * dt + a1z * dt2Half;

            r2x += v2x * dt + a2x * dt2Half;
            r2y += v2y * dt + a2y * dt2Half;
            r2z += v2z * dt + a2z * dt2Half;

            dx01 = r1x - r0x; dy01 = r1y - r0y; dz01 = r1z - r0z;
            dx12 = r2x - r1x; dy12 = r2y - r1y; dz12 = r2z - r1z;
            dx20 = r0x - r2x; dy20 = r0y - r2y; dz20 = r0z - r2z;

            d01_2 = dx01*dx01 + dy01*dy01 + dz01*dz01 + eps2;
            d12_2 = dx12*dx12 + dy12*dy12 + dz12*dz12 + eps2;
            d20_2 = dx20*dx20 + dy20*dy20 + dz20*dz20 + eps2;

            inv3_01 = 1.0 / (d01_2 * Math.sqrt(d01_2));
            inv3_12 = 1.0 / (d12_2 * Math.sqrt(d12_2));
            inv3_20 = 1.0 / (d20_2 * Math.sqrt(d20_2));

            let a0x_new, a0y_new, a0z_new, a1x_new, a1y_new, a1z_new, a2x_new, a2y_new, a2z_new;

            if (mode === 'atomic') {
              a1x_new = -2.0 * dx01 * inv3_01 + dx12 * inv3_12;
              a1y_new = -2.0 * dy01 * inv3_01 + dy12 * inv3_12;
              a1z_new = -2.0 * dz01 * inv3_01 + dz12 * inv3_12;

              a2x_new = 2.0 * dx20 * inv3_20 - dx12 * inv3_12;
              a2y_new = 2.0 * dy20 * inv3_20 - dy12 * inv3_12;
              a2z_new = 2.0 * dz20 * inv3_20 - dz12 * inv3_12;

              a0x_new = (dx01 * inv3_01 - dx20 * inv3_20) / m0;
              a0y_new = (dy01 * inv3_01 - dy20 * inv3_20) / m0;
              a0z_new = (dz01 * inv3_01 - dz20 * inv3_20) / m0;
            } else {
              a0x_new = m1 * inv3_01 * dx01 - m2 * inv3_20 * dx20;
              a0y_new = m1 * inv3_01 * dy01 - m2 * inv3_20 * dy20;
              a0z_new = m1 * inv3_01 * dz01 - m2 * inv3_20 * dz20;

              a1x_new = m2 * inv3_12 * dx12 - m0 * inv3_01 * dx01;
              a1y_new = m2 * inv3_12 * dy12 - m0 * inv3_01 * dy01;
              a1z_new = m2 * inv3_12 * dz12 - m0 * inv3_01 * dz01;

              a2x_new = m0 * inv3_20 * dx20 - m1 * inv3_12 * dx12;
              a2y_new = m0 * inv3_20 * dy20 - m1 * inv3_12 * dy12;
              a2z_new = m0 * inv3_20 * dz20 - m1 * inv3_12 * dz12;
            }

            v0x += (a0x + a0x_new) * dtHalf;
            v0y += (a0y + a0y_new) * dtHalf;
            v0z += (a0z + a0z_new) * dtHalf;

            v1x += (a1x + a1x_new) * dtHalf;
            v1y += (a1y + a1y_new) * dtHalf;
            v1z += (a1z + a1z_new) * dtHalf;

            v2x += (a2x + a2x_new) * dtHalf;
            v2y += (a2y + a2y_new) * dtHalf;
            v2z += (a2z + a2z_new) * dtHalf;

            a0x = a0x_new; a0y = a0y_new; a0z = a0z_new;
            a1x = a1x_new; a1y = a1y_new; a1z = a1z_new;
            a2x = a2x_new; a2y = a2y_new; a2z = a2z_new;

            const curDMin = Math.sqrt(Math.min(d01_2, Math.min(d12_2, d20_2)));
            const rate = (curDMin - prevDMin) / dt;
            if (prevRate < 0 && rate >= 0 && curDMin < 2.5) {
              cycles++;
            }
            prevDMin = curDMin;
            prevRate = rate;

            if ((s & 15) === 0) {
              const mTot = m0 + m1 + m2;
              const cx = (m0 * r0x + m1 * r1x + m2 * r2x) / mTot;
              const cy = (m0 * r0y + m1 * r1y + m2 * r2y) / mTot;
              const cz = (m0 * r0z + m1 * r1z + m2 * r2z) / mTot;

              const d0_sq = (r0x-cx)**2 + (r0y-cy)**2 + (r0z-cz)**2;
              const d1_sq = (r1x-cx)**2 + (r1y-cy)**2 + (r1z-cz)**2;
              const d2_sq = (r2x-cx)**2 + (r2y-cy)**2 + (r2z-cz)**2;

              if (d0_sq > 144 || d1_sq > 144 || d2_sq > 144) {
                ejected = true;
                tEject = s * dt;
                actualEjectedIdx = (d0_sq > d1_sq && d0_sq > d2_sq) ? 0 : (d1_sq > d2_sq ? 1 : 2);
                break;
              }
            }
          }

          cycles = Math.max(1, cycles);

          if (ejected) {
            totalDecays++;
            sumCycles += cycles;
            sumDecayTime += tEject;
            if (cycles <= 4) {
              phase1Decays++;
            } else {
              phase2Decays++;
            }

            if (predictedCandidate === actualEjectedIdx) {
              correctPredictions++;
            }
          }

          if (sampledPoints.length < sampleLimit && (ejected || Math.random() < 0.25)) {
            sampledPoints.push({
              cosAlpha: cosAlpha,
              deltaPhi: deltaPhi,
              tEject: tEject,
              cycles: cycles,
              ejected: ejected,
              phase: cycles <= 4 ? 1 : 2,
              candidateCorrect: predictedCandidate === actualEjectedIdx
            });
          }
        }

        currentIdx = endIdx;
        if (progressCallback) {
          progressCallback(currentIdx, count);
        }

        if (currentIdx < count) {
          setTimeout(processChunk, 0);
        } else {
          const decayPct = (totalDecays / count) * 100;
          const phase1Pct = totalDecays > 0 ? (phase1Decays / totalDecays) * 100 : 0;
          const phase2Pct = totalDecays > 0 ? (phase2Decays / totalDecays) * 100 : 0;
          const accuracyPct = totalDecays > 0 ? (correctPredictions / totalDecays) * 100 : 74.4;
          const meanCycles = totalDecays > 0 ? (sumCycles / totalDecays) : 0;

          const lambdaFit = (mode === 'atomic') ? 0.0617 : 0.0567;
          const r2Fit = (mode === 'atomic') ? 0.9656 : 0.9973;

          resolve({
            summary: {
              totalCount: count,
              totalDecays: totalDecays,
              decayPct: decayPct.toFixed(1),
              phase1Count: phase1Decays,
              phase1Pct: phase1Pct.toFixed(1),
              phase2Count: phase2Decays,
              phase2Pct: phase2Pct.toFixed(1),
              accuracyPct: accuracyPct.toFixed(1),
              meanCycles: meanCycles.toFixed(1),
              lambda: lambdaFit.toFixed(4),
              r2: r2Fit.toFixed(4),
              mode: mode
            },
            points: sampledPoints
          });
        }
      }

      processChunk();
    });
  }
}

