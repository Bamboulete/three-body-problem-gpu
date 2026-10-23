/**
 * Trinitäts-Topologie: Hauptapplikation (App Controller)
 * Orchestriert Benutzeroberfläche, Physik-Engine, 3D-Renderer
 * und das Echtzeit-Oszilloskop für den Ω-Spannungsoperator.
 */

import { PhysicsEngine } from './physics.js';
import { PRESETS } from './presets.js';
import { SceneRenderer } from './renderer.js';

class TrinityApp {
  constructor() {
    this.canvasContainer = document.getElementById('canvas-container');
    this.physics = new PhysicsEngine();
    this.renderer = new SceneRenderer(this.canvasContainer);

    this.isPlaying = true;
    this.speedMultiplier = 1.0;
    this.currentPresetKey = 'figure8';

    // Oszilloskop Puffer
    this.historyLength = 120;
    this.historyData = {
      omega: new Array(this.historyLength).fill(0),
      a1: new Array(this.historyLength).fill(0),
      a2: new Array(this.historyLength).fill(0),
      a3: new Array(this.historyLength).fill(0),
    };

    // FPS & Zeitmessung
    this.lastFrameTime = performance.now();
    this.fpsHistory = [];
    this.frameCount = 0;

    this.initDOMReferences();
    this.bindEvents();
    this.loadPreset('figure8');
    this.startLoop();
  }

  initDOMReferences() {
    // Badges
    this.badgeFps = document.getElementById('badge-fps');
    this.badgeSimTime = document.getElementById('badge-sim-time');

    // Telemetrie
    this.omegaCard = document.getElementById('omega-card');
    this.omegaValue = document.getElementById('omega-value');
    this.omegaRate = document.getElementById('omega-rate');
    this.tensionState = document.getElementById('tension-state');
    this.tensionProgressBar = document.getElementById('tension-progress-bar');

    this.valA1 = document.getElementById('val-a1');
    this.valA2 = document.getElementById('val-a2');
    this.valA3 = document.getElementById('val-a3');
    this.barA1 = document.getElementById('bar-a1');
    this.barA2 = document.getElementById('bar-a2');
    this.barA3 = document.getElementById('bar-a3');

    this.valArea = document.getElementById('val-area');
    this.valSymmetry = document.getElementById('val-symmetry');

    this.valEnergyTot = document.getElementById('val-energy-tot');
    this.valEnergySplit = document.getElementById('val-energy-split');
    this.valEnergyDrift = document.getElementById('val-energy-drift');
    this.valMomentum = document.getElementById('val-momentum');

    // Controls
    this.btnPlayPause = document.getElementById('btn-play-pause');
    this.playIcon = document.getElementById('play-icon');
    this.btnStepForward = document.getElementById('btn-step-forward');
    this.btnReset = document.getElementById('btn-reset');

    this.sliderSpeed = document.getElementById('slider-speed');
    this.labelSpeed = document.getElementById('label-speed');

    this.sliderSoftening = document.getElementById('slider-softening');
    this.labelSoftening = document.getElementById('label-softening');

    this.chkTrails = document.getElementById('chk-trails');
    this.chkTriangleMesh = document.getElementById('chk-triangle-mesh');
    this.chkVectors = document.getElementById('chk-vectors');

    // Massen
    this.sliderMassA = document.getElementById('slider-mass-a');
    this.sliderMassB = document.getElementById('slider-mass-b');
    this.sliderMassC = document.getElementById('slider-mass-c');
    this.labelMassA = document.getElementById('label-mass-a');
    this.labelMassB = document.getElementById('label-mass-b');
    this.labelMassC = document.getElementById('label-mass-c');

    // View Mode
    this.btnViewLab = document.getElementById('btn-view-lab');
    this.btnViewShape = document.getElementById('btn-view-shape');
    this.viewHelpText = document.getElementById('view-help-text');

    // Hypothesen Banner
    this.hypothesisText = document.getElementById('hypothesis-text');

    // Stabilitäts-Filter
    this.stabilityCard = document.getElementById('stability-card');
    this.stabilityBadge = document.getElementById('stability-badge');
    this.stabilityStatusText = document.getElementById('stability-status-text');
    this.stabilityDescText = document.getElementById('stability-desc-text');

    // Multi-Eigenzeit DOM
    this.properTimeCard = document.getElementById('proper-time-card');
    this.riftBadge = document.getElementById('rift-badge');
    this.valTauA = document.getElementById('val-tau-a');
    this.valTauB = document.getElementById('val-tau-b');
    this.valTauC = document.getElementById('val-tau-c');
    this.valRateA = document.getElementById('val-rate-a');
    this.valRateB = document.getElementById('val-rate-b');
    this.valRateC = document.getElementById('val-rate-c');
    this.riftStatusDesc = document.getElementById('rift-status-desc');

    // Pfad 3: Analytische Formel & Auswurf-Prädiktor DOM
    this.formulaCard = document.getElementById('formula-card');
    this.formulaBadge = document.getElementById('formula-badge');
    this.predTarget = document.getElementById('pred-target');
    this.predConf = document.getElementById('pred-conf');
    this.predStageLabel = document.getElementById('pred-stage-label');
    this.predTime = document.getElementById('pred-time');
    this.trackerStatus = document.getElementById('tracker-status');
    this.trackerPct = document.getElementById('tracker-pct');
    this.trackerBarFill = document.getElementById('tracker-bar-fill');
    this.valRotBonus = document.getElementById('val-rot-bonus');
    this.valSlingshotTime = document.getElementById('val-slingshot-time');
    this.valCyclesCount = document.getElementById('val-cycles-count');
    this.valPhaseIntegral = document.getElementById('val-phase-integral');
    this.valDopplerGate = document.getElementById('val-doppler-gate');

    // Formel Klartext / Mathe Umschalter
    this.btnFmodePlain = document.getElementById('btn-fmode-plain');
    this.btnFmodeMath = document.getElementById('btn-fmode-math');
    this.formulaPlainView = document.getElementById('formula-plain-view');
    this.formulaMathView = document.getElementById('formula-math-view');
    this.plainNumbersRow = document.getElementById('plain-numbers-row');

    // Chart Tabs & Mode
    this.chartMode = 'time'; // 'time' oder 'phase'
    this.chartTitle = document.getElementById('chart-title');
    this.tabChartTime = document.getElementById('tab-chart-time');
    this.tabChartPhase = document.getElementById('tab-chart-phase');
    this.chartLegend = document.getElementById('chart-legend');

    // Canvas Oszilloskop
    this.oscCanvas = document.getElementById('oscilloscope-canvas');
    this.oscCtx = this.oscCanvas.getContext('2d');
  }

  bindEvents() {
    // Presets Buttons
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const targetBtn = e.currentTarget;
        const presetKey = targetBtn.dataset.preset;
        if (PRESETS[presetKey]) {
          presetButtons.forEach(b => b.classList.remove('active'));
          targetBtn.classList.add('active');
          this.loadPreset(presetKey);
        }
      });
    });

    // Play/Pause
    this.btnPlayPause.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      this.playIcon.textContent = this.isPlaying ? '⏸️' : '▶️';
      this.btnPlayPause.classList.toggle('primary', this.isPlaying);
    });

    // Step Forward
    this.btnStepForward.addEventListener('click', () => {
      this.isPlaying = false;
      this.playIcon.textContent = '▶️';
      this.btnPlayPause.classList.remove('primary');
      this.physics.update(0.016, 1.0);
      this.renderer.update(this.physics, this.frameCount);
      this.updateTelemetry();
    });

    // Reset
    this.btnReset.addEventListener('click', () => {
      this.loadPreset(this.currentPresetKey);
    });

    // Speed Slider
    this.sliderSpeed.addEventListener('input', (e) => {
      this.speedMultiplier = parseFloat(e.target.value);
      this.labelSpeed.textContent = `${this.speedMultiplier.toFixed(1)}x`;
    });

    // Softening Slider
    this.sliderSoftening.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.physics.softening = val;
      this.labelSoftening.textContent = val.toFixed(3);
    });

    // Toggles
    this.chkTrails.addEventListener('change', (e) => {
      this.renderer.showTrails = e.target.checked;
    });

    this.chkTriangleMesh.addEventListener('change', (e) => {
      this.renderer.showTriangleMesh = e.target.checked;
    });

    this.chkVectors.addEventListener('change', (e) => {
      this.renderer.showVectors = e.target.checked;
    });

    // Mass Sliders
    const updateMass = (idx, val, labelElem) => {
      if (this.physics.bodies[idx]) {
        this.physics.bodies[idx].mass = val;
        labelElem.textContent = val.toFixed(1);
      }
    };

    this.sliderMassA.addEventListener('input', (e) => {
      updateMass(0, parseFloat(e.target.value), this.labelMassA);
    });
    this.sliderMassB.addEventListener('input', (e) => {
      updateMass(1, parseFloat(e.target.value), this.labelMassB);
    });
    this.sliderMassC.addEventListener('input', (e) => {
      updateMass(2, parseFloat(e.target.value), this.labelMassC);
    });

    // View Modes (Klassischer Raum vs Trinitäts-Formraum)
    this.btnViewLab.addEventListener('click', () => {
      this.btnViewLab.classList.add('active');
      this.btnViewShape.classList.remove('active');
      this.renderer.setViewMode('lab');
      this.viewHelpText.textContent = 'Klassischer Raum: Du beobachtest die drei Körper im übergeordneten Koordinatensystem.';
    });

    this.btnViewShape.addEventListener('click', () => {
      this.btnViewShape.classList.add('active');
      this.btnViewLab.classList.remove('active');
      this.renderer.setViewMode('shape');
      this.viewHelpText.textContent = 'Trinitäts-Formraum: Kamera folgt dem Schwerpunkt und Dreieck. Das System rotiert nicht mehr – du siehst die reine Form-Dehnung!';
    });

    // Chart Mode Tabs (t-Welle vs Phasenraum)
    this.tabChartTime.addEventListener('click', () => {
      this.chartMode = 'time';
      this.tabChartTime.classList.add('active');
      this.tabChartPhase.classList.remove('active');
      this.chartTitle.textContent = 'Ω-Oszilloskop';
      this.chartLegend.innerHTML = `
        <span class="legend-item" style="color: #ff9e00;">— Ω Spannung</span>
        <span class="legend-item" style="color: #00f0ff;">— a₁</span>
        <span class="legend-item" style="color: #ff007f;">— a₂</span>
        <span class="legend-item" style="color: #00ff88;">— a₃</span>
      `;
      this.drawOscilloscope();
    });

    this.tabChartPhase.addEventListener('click', () => {
      this.chartMode = 'phase';
      this.tabChartPhase.classList.add('active');
      this.tabChartTime.classList.remove('active');
      this.chartTitle.textContent = 'Phasenraum [Ω vs dΩ/dt]';
      this.chartLegend.innerHTML = `
        <span class="legend-item" style="color: #00ff88;">— Grenzzyklus / Orbit</span>
        <span class="legend-item" style="color: #8e9bb4;">X: Ω | Y: dΩ/dt</span>
      `;
      this.drawPhaseSpace();
    });

    // Formel Klartext vs Mathe Modus Umschaltung
    if (this.btnFmodePlain && this.btnFmodeMath) {
      this.btnFmodePlain.addEventListener('click', () => {
        this.btnFmodePlain.classList.add('active');
        this.btnFmodeMath.classList.remove('active');
        this.formulaPlainView.style.display = 'flex';
        this.formulaMathView.style.display = 'none';
      });

      this.btnFmodeMath.addEventListener('click', () => {
        this.btnFmodeMath.classList.add('active');
        this.btnFmodePlain.classList.remove('active');
        this.formulaMathView.style.display = 'block';
        this.formulaPlainView.style.display = 'none';
      });
    }
  }

  loadPreset(presetKey) {
    const preset = PRESETS[presetKey];
    if (!preset) return;

    this.currentPresetKey = presetKey;
    this.physics.softening = preset.softening;
    this.sliderSoftening.value = preset.softening;
    this.labelSoftening.textContent = preset.softening.toFixed(3);

    const bodies = preset.bodies();
    this.physics.setBodies(bodies);
    this.renderer.clearTrails();
    this.renderer.setCameraDistance(preset.cameraDist);

    // Reset Oszilloskop-Daten
    const initialOmega = this.physics.calculateOmega();
    const sides = this.physics.getTriangleSides();
    this.historyData.omega.fill(initialOmega);
    this.historyData.a1.fill(sides.a1);
    this.historyData.a2.fill(sides.a2);
    this.historyData.a3.fill(sides.a3);

    // Massen Regler synchronisieren
    this.sliderMassA.value = bodies[0].mass;
    this.labelMassA.textContent = bodies[0].mass.toFixed(1);
    this.sliderMassB.value = bodies[1].mass;
    this.labelMassB.textContent = bodies[1].mass.toFixed(1);
    this.sliderMassC.value = bodies[2].mass;
    this.labelMassC.textContent = bodies[2].mass.toFixed(1);

    // Hypothesen Banner anpassen
    this.hypothesisText.innerHTML = `<strong>Szenario [${preset.name}]:</strong> ${preset.description}`;
  }

  updateTelemetry() {
    const omega = this.physics.calculateOmega();
    const omegaRate = this.physics.omegaRate;
    const sides = this.physics.getTriangleSides();
    const area = this.physics.calculateTriangleArea();
    const symmetry = this.physics.calculateSymmetryIndex();

    // Omega Hero Display
    this.omegaValue.textContent = omega.toFixed(3);
    this.omegaRate.textContent = `dΩ/dt: ${omegaRate >= 0 ? '+' : ''}${omegaRate.toFixed(2)}`;

    // Tension State & Styling
    let stateLabel = 'Gleichgewicht';
    let tensionClass = '';
    let barPercent = Math.min(100, Math.max(5, (omega / 4.0) * 100));

    if (omega > 5.0 || Math.abs(omegaRate) > 12.0) {
      stateLabel = 'Kritische Nahbegegnung';
      tensionClass = 'critical-tension';
    } else if (omega > 2.5 || Math.abs(omegaRate) > 4.0) {
      stateLabel = 'Starke Deformation';
    } else if (symmetry > 88) {
      stateLabel = 'Symmetrische Harmonie';
    }

    this.tensionState.textContent = stateLabel;
    this.tensionProgressBar.style.width = `${barPercent}%`;

    if (tensionClass) {
      this.omegaCard.classList.add('critical-tension');
    } else {
      this.omegaCard.classList.remove('critical-tension');
    }

    // Seitenlängen
    this.valA1.textContent = `${sides.a1.toFixed(2)} AU`;
    this.valA2.textContent = `${sides.a2.toFixed(2)} AU`;
    this.valA3.textContent = `${sides.a3.toFixed(2)} AU`;

    const maxDist = Math.max(sides.a1, sides.a2, sides.a3, 4.0);
    this.barA1.style.width = `${Math.min(100, (sides.a1 / maxDist) * 100)}%`;
    this.barA2.style.width = `${Math.min(100, (sides.a2 / maxDist) * 100)}%`;
    this.barA3.style.width = `${Math.min(100, (sides.a3 / maxDist) * 100)}%`;

    // Fläche & Symmetrie
    this.valArea.textContent = `${area.toFixed(2)} AU²`;
    this.valSymmetry.textContent = `${symmetry}%`;

    // Erhaltungsgrößen
    const totalE = this.physics.calculateTotalEnergy();
    const kinE = this.physics.calculateKineticEnergy();
    const potE = this.physics.calculatePotentialEnergy();
    const momL = this.physics.calculateAngularMomentum();

    this.valEnergyTot.textContent = totalE.toFixed(3);
    this.valEnergySplit.textContent = `${kinE.toFixed(2)} / ${potE.toFixed(2)}`;

    if (this.physics.initialEnergy !== null && Math.abs(this.physics.initialEnergy) > 1e-5) {
      const drift = ((totalE - this.physics.initialEnergy) / Math.abs(this.physics.initialEnergy)) * 100;
      this.valEnergyDrift.textContent = `${drift >= 0 ? '+' : ''}${drift.toFixed(3)}%`;
      this.valEnergyDrift.style.color = Math.abs(drift) < 0.5 ? 'var(--accent-green)' : '#ffb703';
    }

    this.valMomentum.textContent = momL.toFixed(3);

    // Topologischer Stabilitäts-Filter auswerten
    if (this.frameCount % 5 === 0) {
      const stability = this.physics.evaluateStability();
      this.stabilityBadge.textContent = `${stability.score}% ${stability.state === 'PERIODISCH' ? 'STABIL' : stability.state}`;
      this.stabilityBadge.style.color = stability.color;
      this.stabilityBadge.style.borderColor = `${stability.color}66`;
      this.stabilityBadge.style.background = `${stability.color}1c`;
      this.stabilityStatusText.textContent = stability.label;
      this.stabilityStatusText.style.color = stability.color;
      this.stabilityDescText.textContent = stability.desc || '';
    }

    // Multi-Eigenzeit Daten (Deine Hypothese)
    const ptData = this.physics.getProperTimeData();
    this.valTauA.textContent = `${ptData.tauA.toFixed(2)}s`;
    this.valTauB.textContent = `${ptData.tauB.toFixed(2)}s`;
    this.valTauC.textContent = `${ptData.tauC.toFixed(2)}s`;
    this.valRateA.textContent = `${ptData.rateA.toFixed(2)}x`;
    this.valRateB.textContent = `${ptData.rateB.toFixed(2)}x`;
    this.valRateC.textContent = `${ptData.rateC.toFixed(2)}x`;

    this.riftBadge.textContent = `Δτ: ${ptData.maxRift.toFixed(3)}s`;

    if (ptData.maxRift > 0.35) {
      this.riftBadge.style.color = '#ff0055';
      this.riftBadge.style.background = 'rgba(255, 0, 85, 0.2)';
      this.riftBadge.style.borderColor = '#ff0055';
      this.properTimeCard.classList.add('rift-warning');
      this.riftStatusDesc.textContent = '⚠️ Kritischer Zeit-Riss! Ein Körper entkoppelt sich aus dem Zeitgefüge (Phasen-Abriss).';
      this.riftStatusDesc.style.color = '#ff0055';
    } else if (ptData.maxRift > 0.12) {
      this.riftBadge.style.color = '#ffb703';
      this.riftBadge.style.background = 'rgba(255, 183, 3, 0.2)';
      this.riftBadge.style.borderColor = '#ffb703';
      this.properTimeCard.classList.remove('rift-warning');
      this.riftStatusDesc.textContent = 'Zeit-Schere öffnet sich: Uhren driften asymmetrisch auseinander.';
      this.riftStatusDesc.style.color = '#ffb703';
    } else {
      this.riftBadge.style.color = 'var(--color-body-a)';
      this.riftBadge.style.background = 'rgba(0, 240, 255, 0.15)';
      this.riftBadge.style.borderColor = 'rgba(0, 240, 255, 0.35)';
      this.properTimeCard.classList.remove('rift-warning');
      this.riftStatusDesc.textContent = 'Zeittakt synchron: Harmonische Phasenkopplung.';
      this.riftStatusDesc.style.color = 'var(--text-muted)';
    }

    // Pfad 3: Analytische Auswurf-Formel & Prädiktor aktualisieren
    const formulaPred = this.physics.predictEjectionFormula();
    const stability = this.physics.evaluateStability();

    if (formulaPred && this.formulaCard) {
      if (stability.state === 'PERIODISCH') {
        this.predTarget.textContent = 'Kein Zerfall';
        this.predTarget.style.color = '#00ff88';
        this.predConf.textContent = 'Stabil';
        this.predConf.style.color = '#00ff88';
        this.predConf.style.background = 'rgba(0, 255, 136, 0.15)';
        this.predTime.textContent = '∞ (Stabil)';
        this.trackerStatus.textContent = 'Grenzzyklus:';
        this.trackerPct.textContent = '100% Sicher';
        this.trackerBarFill.style.width = '0%';
        this.trackerBarFill.style.background = 'linear-gradient(90deg, #00ff88, #00f0ff)';
      } else if (stability.state === 'AUSWURF') {
        this.predTarget.textContent = `${formulaPred.candidateName} fliegt!`;
        this.predTarget.style.color = '#ff0055';
        this.predConf.textContent = 'AUSWURF';
        this.predConf.style.color = '#ff0055';
        this.predConf.style.background = 'rgba(255, 0, 85, 0.2)';
        this.predTime.textContent = `${this.physics.time.toFixed(1)} s (Real)`;
        this.trackerStatus.textContent = 'System zerbrochen:';
        this.trackerPct.textContent = '100%';
        this.trackerBarFill.style.width = '100%';
        this.trackerBarFill.style.background = '#ff0055';
      } else {
        // Chaotisch oder Metastabil: Analytische Vorhersage aktiv
        const t_pred = formulaPred.t_pred_symbolic;
        const progress = Math.min(1.0, this.physics.time / t_pred);
        const pct = Math.round(progress * 100);

        if (this.predStageLabel) {
          this.predStageLabel.textContent = formulaPred.isSlingshotLocked 
            ? 'Auswurf-Kandidat (Stufe 2):' 
            : 'Auswurf-Kandidat (Stufe 1):';
        }

        this.predTarget.textContent = formulaPred.candidateName;
        this.predTarget.style.color = formulaPred.candidateColor;

        if (formulaPred.isSlingshotLocked) {
          this.predConf.textContent = '100% Sicher';
          this.predConf.style.color = '#00ff88';
          this.predConf.style.background = 'rgba(0, 255, 136, 0.2)';
          if (formulaPred.exact_eject_time) {
            this.predTime.textContent = `${formulaPred.exact_eject_time.toFixed(1)} s (Exakt)`;
          } else {
            this.predTime.textContent = `~${t_pred.toFixed(1)} s`;
          }
        } else {
          this.predConf.textContent = `${formulaPred.confidencePct}%`;
          this.predConf.style.color = formulaPred.candidateColor;
          this.predConf.style.background = 'rgba(255, 0, 127, 0.15)';
          this.predTime.textContent = `~${t_pred.toFixed(1)} s`;
        }

        if (this.valRotBonus) {
          this.valRotBonus.textContent = `+${formulaPred.rot_bonus_pct}%`;
        }
        if (this.valDopplerGate) {
          const isSturz = formulaPred.cos_alpha < -0.05;
          const label = isSturz ? 'Sturzflug' : 'Resonanz';
          const color = isSturz ? '#ff0055' : '#00ff88';
          this.valDopplerGate.textContent = `${label} (${formulaPred.cos_alpha >= 0 ? '+' : ''}${formulaPred.cos_alpha.toFixed(2)})`;
          this.valDopplerGate.style.color = color;
        }
        if (this.valSlingshotTime) {
          if (formulaPred.isSlingshotLocked) {
            this.valSlingshotTime.textContent = `${this.physics.slingshotTime.toFixed(1)}s (Riss!)`;
          } else {
            this.valSlingshotTime.textContent = `~${formulaPred.t_slingshot.toFixed(1)}s`;
          }
        }
        if (this.valCyclesCount) {
          const cycText = formulaPred.isSlingshotLocked 
            ? `${formulaPred.cycles_count} (Riss)` 
            : `${formulaPred.cycles_count} / ~${formulaPred.n_pred}`;
          this.valCyclesCount.textContent = cycText;
        }
        if (this.valPhaseIntegral) {
          this.valPhaseIntegral.textContent = `Σ = ${formulaPred.sigma_integral.toFixed(1)}`;
        }

        if (this.plainNumbersRow) {
          const dopplerSign = formulaPred.cos_alpha < -0.05 ? 'Sturzflug' : 'Resonanz';
          this.plainNumbersRow.innerHTML = `
            <span class="font-mono text-cyan" style="font-weight: 700;">${t_pred.toFixed(1)}s</span> = 
            (<span class="font-mono text-green">${formulaPred.n_pred} Zyklen</span> − 1) × 
            <span class="font-mono">${formulaPred.t_orbit.toFixed(1)}s</span> + 
            <span class="font-mono text-gold">${formulaPred.t_kepler.toFixed(1)}s</span>
            <span class="text-muted" style="font-size: 0.70rem; margin-left: 6px;">[${dopplerSign}, cos α: ${formulaPred.cos_alpha.toFixed(2)}]</span>
          `;
        }

        const remaining = Math.max(0, t_pred - this.physics.time);
        this.trackerStatus.textContent = `Countdown (${remaining.toFixed(1)}s):`;
        this.trackerPct.textContent = `${pct}%`;
        this.trackerBarFill.style.width = `${pct}%`;
        this.trackerBarFill.style.background = 'linear-gradient(90deg, #00f0ff, #ffb703, #ff0055)';
      }
    }

    // Oszilloskop Puffer aktualisieren
    if (this.frameCount % 2 === 0) {
      this.historyData.omega.shift();
      this.historyData.omega.push(omega);

      this.historyData.a1.shift();
      this.historyData.a1.push(sides.a1);

      this.historyData.a2.shift();
      this.historyData.a2.push(sides.a2);

      this.historyData.a3.shift();
      this.historyData.a3.push(sides.a3);

      if (this.chartMode === 'phase') {
        this.drawPhaseSpace();
      } else {
        this.drawOscilloscope();
      }
    }
  }

  drawOscilloscope() {
    const ctx = this.oscCtx;
    const w = this.oscCanvas.width;
    const h = this.oscCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Subtiles Hintergrundgitter
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.25); ctx.lineTo(w, h * 0.25);
    ctx.moveTo(0, h * 0.5);  ctx.lineTo(w, h * 0.5);
    ctx.moveTo(0, h * 0.75); ctx.lineTo(w, h * 0.75);
    ctx.stroke();

    // Kurven zeichnen
    const drawLine = (arr, color, maxScale, lineW = 1.5) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineW;
      ctx.beginPath();
      const n = arr.length;
      const step = w / (n - 1);

      for (let i = 0; i < n; i++) {
        const val = Math.max(0, Math.min(maxScale, arr[i]));
        const y = h - (val / maxScale) * (h - 8) - 4;
        const x = i * step;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    // Sanfte Skalen-Dämpfung gegen sprunghafte Vergrößerungen
    if (!this.smoothMaxA) this.smoothMaxA = 5.0;
    if (!this.smoothMaxO) this.smoothMaxO = 6.0;

    const targetMaxA = Math.max(5.0, ...this.historyData.a1, ...this.historyData.a2, ...this.historyData.a3);
    const targetMaxO = Math.max(6.0, ...this.historyData.omega);

    this.smoothMaxA += (targetMaxA - this.smoothMaxA) * 0.05;
    this.smoothMaxO += (targetMaxO - this.smoothMaxO) * 0.05;

    drawLine(this.historyData.a1, 'rgba(0, 240, 255, 0.65)', this.smoothMaxA, 1.2);
    drawLine(this.historyData.a2, 'rgba(255, 0, 127, 0.65)', this.smoothMaxA, 1.2);
    drawLine(this.historyData.a3, 'rgba(0, 255, 136, 0.65)', this.smoothMaxA, 1.2);
    drawLine(this.historyData.omega, '#ff9e00', this.smoothMaxO, 2.0); // Dicke hervorgehobene Ω Kurve
  }

  // Phasenraum: Visualisiert den Grenzzyklus (Ω vs dΩ/dt)
  drawPhaseSpace() {
    const ctx = this.oscCtx;
    const w = this.oscCanvas.width;
    const h = this.oscCanvas.height;

    ctx.clearRect(0, 0, w, h);

    const history = this.physics.phaseHistory;
    if (!history || history.length < 5) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('Phasenraum wird aufgebaut...', w * 0.5, h * 0.5);
      return;
    }

    // Fadenkreuz in der Mitte
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 0); ctx.lineTo(w * 0.5, h);
    ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    // Dynamische Skalierung
    let minO = Infinity, maxO = -Infinity;
    let maxRate = 1.0;
    for (const pt of history) {
      if (pt.omega < minO) minO = pt.omega;
      if (pt.omega > maxO) maxO = pt.omega;
      if (Math.abs(pt.dOmega) > maxRate) maxRate = Math.abs(pt.dOmega);
    }
    const rangeO = Math.max(0.5, maxO - minO);
    const midO = (minO + maxO) / 2.0;

    // Phasenkurve zeichnen (Grenzzyklus)
    ctx.lineWidth = 1.6;
    for (let i = 1; i < history.length; i++) {
      const p0 = history[i - 1];
      const p1 = history[i];

      const x0 = w * 0.5 + ((p0.omega - midO) / rangeO) * (w * 0.85);
      const y0 = h * 0.5 - (p0.dOmega / maxRate) * (h * 0.42);

      const x1 = w * 0.5 + ((p1.omega - midO) / rangeO) * (w * 0.85);
      const y1 = h * 0.5 - (p1.dOmega / maxRate) * (h * 0.42);

      const alpha = (i / history.length) * 0.85 + 0.15;
      ctx.strokeStyle = `rgba(0, 255, 136, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // Aktueller Zustandspunkt
    const last = history[history.length - 1];
    const currX = w * 0.5 + ((last.omega - midO) / rangeO) * (w * 0.85);
    const currY = h * 0.5 - (last.dOmega / maxRate) * (h * 0.42);

    ctx.fillStyle = '#ff9e00';
    ctx.beginPath();
    ctx.arc(currX, currY, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  startLoop() {
    const loop = (currentTime) => {
      requestAnimationFrame(loop);

      const delta = (currentTime - this.lastFrameTime) / 1000.0;
      this.lastFrameTime = currentTime;
      this.frameCount++;

      // FPS Tracking
      if (delta > 0) {
        const currentFps = Math.round(1.0 / delta);
        this.fpsHistory.push(currentFps);
        if (this.fpsHistory.length > 25) this.fpsHistory.shift();

        if (this.frameCount % 15 === 0) {
          const avgFps = Math.round(this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length);
          this.badgeFps.textContent = `${avgFps} FPS`;
          this.badgeSimTime.textContent = `t = ${this.physics.time.toFixed(1)} s`;
        }
      }

      if (this.isPlaying) {
        this.physics.update(delta, this.speedMultiplier);
      }

      this.renderer.update(this.physics, this.frameCount);
      this.updateTelemetry();
    };

    requestAnimationFrame(loop);
  }
}

// Start beim Laden des Fensters
window.addEventListener('DOMContentLoaded', () => {
  new TrinityApp();
});
