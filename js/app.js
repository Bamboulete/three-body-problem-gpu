/**
 * Trinitäts-Topologie: Hauptapplikation (App Controller)
 * Orchestriert Benutzeroberfläche, Physik-Engine (Kosmos & Quanten-Atom),
 * 3D-Renderer, Mission-Control-HUD und das Live-Monte-Carlo-Labor.
 */

import { PhysicsEngine } from './physics.js?v=6';
import { COSMIC_PRESETS, ATOMIC_PRESETS, PRESETS } from './presets.js?v=6';
import { SceneRenderer } from './renderer.js?v=6';

class TrinityApp {
  constructor() {
    window.app = this;
    this.canvasContainer = document.getElementById('canvas-container');
    this.physics = new PhysicsEngine();
    this.renderer = new SceneRenderer(this.canvasContainer);

    this.isPlaying = true;
    this.speedMultiplier = 1.0;
    this.currentMode = 'cosmic'; // 'cosmic' oder 'atomic'
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
    this.initMonteCarloModal();
    this.setPhysicsMode('cosmic');
    this.startLoop();
  }

  initDOMReferences() {
    // Badges & Header
    this.badgeFps = document.getElementById('badge-fps');
    this.badgeSimTime = document.getElementById('badge-sim-time');
    this.headerSubtitle = document.getElementById('header-subtitle');
    this.btnModeCosmic = document.getElementById('btn-mode-cosmic');
    this.btnModeAtomic = document.getElementById('btn-mode-atomic');

    // Floating Mission Control HUD (At-a-Glance Cockpit)
    this.hudStatusBadge = document.getElementById('hud-status-badge');
    this.hudStatusSub = document.getElementById('hud-status-sub');
    this.hudDopplerNeedle = document.getElementById('hud-doppler-needle');
    this.hudDopplerText = document.getElementById('hud-doppler-text');
    this.hudDopplerDesc = document.getElementById('hud-doppler-desc');
    this.hudShearFill = document.getElementById('hud-shear-fill');
    this.hudShearText = document.getElementById('hud-shear-text');
    this.hudShearDesc = document.getElementById('hud-shear-desc');
    this.hudPredTarget = document.getElementById('hud-pred-target');
    this.hudPredConf = document.getElementById('hud-pred-conf');
    this.hudPredDetail = document.getElementById('hud-pred-detail');

    // Left Panel Tabs
    this.tabNavMetrics = document.getElementById('tab-nav-metrics');
    this.tabNavPredictor = document.getElementById('tab-nav-predictor');
    this.tabNavConservation = document.getElementById('tab-nav-conservation');
    this.tabContentMetrics = document.getElementById('tab-content-metrics');
    this.tabContentPredictor = document.getElementById('tab-content-predictor');
    this.tabContentConservation = document.getElementById('tab-content-conservation');

    // Telemetrie (Metriken & GS-Gewebe)
    this.omegaCard = document.getElementById('omega-card');
    this.omegaSym = document.getElementById('omega-sym');
    this.omegaTitle = document.getElementById('omega-title');
    this.omegaValue = document.getElementById('omega-value');
    this.omegaRate = document.getElementById('omega-rate');
    this.omegaDescSub = document.getElementById('omega-desc-sub');
    this.tensionState = document.getElementById('tension-state');
    this.tensionProgressBar = document.getElementById('tension-progress-bar');

    this.labelSideA1 = document.getElementById('label-side-a1');
    this.labelSideA2 = document.getElementById('label-side-a2');
    this.labelSideA3 = document.getElementById('label-side-a3');
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

    // Massen & Ladungen
    this.massesSummary = document.getElementById('masses-summary');
    this.sliderMassA = document.getElementById('slider-mass-a');
    this.sliderMassB = document.getElementById('slider-mass-b');
    this.sliderMassC = document.getElementById('slider-mass-c');
    this.labelBodyA = document.getElementById('label-body-a');
    this.labelBodyB = document.getElementById('label-body-b');
    this.labelBodyC = document.getElementById('label-body-c');
    this.labelMassA = document.getElementById('label-mass-a');
    this.labelMassB = document.getElementById('label-mass-b');
    this.labelMassC = document.getElementById('label-mass-c');

    // View Mode
    this.btnViewLab = document.getElementById('btn-view-lab');
    this.btnViewShape = document.getElementById('btn-view-shape');
    this.viewHelpText = document.getElementById('view-help-text');

    // Hypothesen Banner
    this.hypothesisBadge = document.getElementById('hypothesis-badge');
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
    this.labelTauA = document.getElementById('label-tau-a');
    this.labelTauB = document.getElementById('label-tau-b');
    this.labelTauC = document.getElementById('label-tau-c');
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
    // Mode Switcher Buttons
    if (this.btnModeCosmic) {
      this.btnModeCosmic.addEventListener('click', () => {
        this.setPhysicsMode('cosmic');
      });
    }

    if (this.btnModeAtomic) {
      this.btnModeAtomic.addEventListener('click', () => {
        this.setPhysicsMode('atomic');
      });
    }

    // Left Panel Tab Navigation
    const tabs = [
      { btn: this.tabNavMetrics, pane: this.tabContentMetrics },
      { btn: this.tabNavPredictor, pane: this.tabContentPredictor },
      { btn: this.tabNavConservation, pane: this.tabContentConservation }
    ];

    tabs.forEach(({ btn, pane }) => {
      if (btn) {
        btn.addEventListener('click', () => {
          tabs.forEach(t => {
            if (t.btn) t.btn.classList.remove('active');
            if (t.pane) t.pane.style.display = 'none';
          });
          btn.classList.add('active');
          if (pane) pane.style.display = 'block';
        });
      }
    });

    // Play / Pause
    this.btnPlayPause.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      this.playIcon.textContent = this.isPlaying ? '⏸️' : '▶️';
      this.btnPlayPause.innerHTML = `${this.playIcon.outerHTML} ${this.isPlaying ? 'Pause' : 'Start'}`;
      this.btnPlayPause.classList.toggle('primary', this.isPlaying);
    });

    // Einzelschritt vorwärts
    this.btnStepForward.addEventListener('click', () => {
      this.isPlaying = false;
      this.playIcon.textContent = '▶️';
      this.btnPlayPause.innerHTML = `${this.playIcon.outerHTML} Start`;
      this.btnPlayPause.classList.remove('primary');
      this.physics.update(0.016, 1.0);
      this.renderer.update(this.physics, this.frameCount);
      this.updateTelemetry();
    });

    // Reset Szenario
    this.btnReset.addEventListener('click', () => {
      this.loadPreset(this.currentPresetKey);
    });

    // Geschwindigkeit Slider
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

    // Overlays Checkboxes
    this.chkTrails.addEventListener('change', (e) => {
      this.renderer.setShowTrails(e.target.checked);
    });

    this.chkTriangleMesh.addEventListener('change', (e) => {
      this.renderer.setShowTriangleMesh(e.target.checked);
    });

    this.chkVectors.addEventListener('change', (e) => {
      this.renderer.setShowVectors(e.target.checked);
    });

    // Massen Slider
    const updateMass = (index, value) => {
      if (this.physics.bodies[index]) {
        this.physics.bodies[index].mass = value;
      }
    };

    this.sliderMassA.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.labelMassA.textContent = val.toFixed(1);
      updateMass(0, val);
    });

    this.sliderMassB.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.labelMassB.textContent = val.toFixed(1);
      updateMass(1, val);
    });

    this.sliderMassC.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.labelMassC.textContent = val.toFixed(1);
      updateMass(2, val);
    });

    // Blickwinkel-Modus (Klassischer Raum vs Trinitäts-Formraum)
    this.btnViewLab.addEventListener('click', () => {
      this.btnViewLab.classList.add('active');
      this.btnViewShape.classList.remove('active');
      this.renderer.setViewMode('lab');
      this.viewHelpText.textContent = 'Klassische Ansicht: Du siehst die drei Körper im Koordinatensystem des Raums fliegen.';
    });

    this.btnViewShape.addEventListener('click', () => {
      this.btnViewShape.classList.add('active');
      this.btnViewLab.classList.remove('active');
      this.renderer.setViewMode('shape');
      this.viewHelpText.textContent = 'Trinitäts-Formraum: Kamera rotiert synchron mit dem Dreieck GS. Das Gewebe bleibt statisch im Blickfeld.';
    });

    // Oszilloskop Modus Umschaltung
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

  setPhysicsMode(mode) {
    this.currentMode = mode;
    this.physics.setMode(mode);
    this.renderer.setMode(mode);

    if (mode === 'atomic') {
      this.btnModeAtomic.classList.add('active');
      this.btnModeCosmic.classList.remove('active');
      this.headerSubtitle.innerHTML = 'Helium-3-Körper-Labor &middot; Coulomb-Topologie &middot; Skalen-Universalität';
      this.labelSideA1.textContent = 'Seite a₁ (Kern ↔ e₁):';
      this.labelSideA2.textContent = 'Seite a₂ (e₁ ↔ e₂):';
      this.labelSideA3.textContent = 'Seite a₃ (e₂ ↔ Kern):';
      this.omegaDescSub.textContent = 'Coulomb-Bindung & Abstoßung';
      this.massesSummary.textContent = 'Massen & Ladungen konfigurieren';
      this.labelBodyA.textContent = 'Masse Kern He²⁺ (Gold):';
      this.labelBodyB.textContent = 'Masse Elektron e₁ (Cyan):';
      this.labelBodyC.textContent = 'Masse Elektron e₂ (Magenta):';
      this.labelTauA.innerHTML = '<span class="dot dot-b"></span> τ_Kern (Gold)';
      this.labelTauB.innerHTML = '<span class="dot dot-a"></span> τ_e1 (Cyan)';
      this.labelTauC.innerHTML = '<span class="dot dot-c"></span> τ_e2 (Magenta)';
      this.hypothesisBadge.textContent = 'ATOM-HYPOTHESE';
      this.hypothesisText.innerHTML = '<strong>Helium-Coulomb-Topologie:</strong> Kern He²⁺ mit Coulomb-Abstoßung (+1/r₁₂) zwischen beiden Elektronen. Der Wannier-Sattel (cos α → -1.0) bewirkt sofortige Autoionisation; Frozen Planet (cos α → +1.0) schirmt das Paar phasenverriegelt ab (λ = 0.0617).';
    } else {
      this.btnModeCosmic.classList.add('active');
      this.btnModeAtomic.classList.remove('active');
      this.headerSubtitle.innerHTML = '3-Körper-Labor &middot; Skalen-Universalität (10²⁶ m ↔ 10⁻¹⁰ m)';
      this.labelSideA1.textContent = 'Seite a₁ (Stern A ↔ B):';
      this.labelSideA2.textContent = 'Seite a₂ (Stern B ↔ C):';
      this.labelSideA3.textContent = 'Seite a₃ (Stern C ↔ A):';
      this.omegaDescSub.textContent = 'Innere Feldspannung des Dreiecks';
      this.massesSummary.textContent = 'Massen konfigurieren (mₐ, m_b, m_c)';
      this.labelBodyA.textContent = 'Masse Stern A (Cyan):';
      this.labelBodyB.textContent = 'Masse Stern B (Gold):';
      this.labelBodyC.textContent = 'Masse Stern C (Magenta):';
      this.labelTauA.innerHTML = '<span class="dot dot-a"></span> τₐ (Cyan)';
      this.labelTauB.innerHTML = '<span class="dot dot-b"></span> τ_b (Gold)';
      this.labelTauC.innerHTML = '<span class="dot dot-c"></span> τ_c (Magenta)';
      this.hypothesisBadge.textContent = 'HYPOTHESE';
      this.hypothesisText.innerHTML = '<strong>Trinitäts-Prämisse:</strong> Gravitationskraft als elastisches Spannungsfeld Ω im Deformationsraum GS. Phase 1 (N≤4) deterministisch, Phase 2 (N≥5) ergodischer Poisson-Zerfall (λ = 0.0567).';
    }

    this.renderPresetButtons();
    const defaultPreset = (mode === 'atomic') ? 'wannier' : 'figure8';
    this.loadPreset(defaultPreset);
  }

  renderPresetButtons() {
    const container = document.getElementById('preset-buttons-container');
    if (!container) return;
    container.innerHTML = '';

    const presetMap = (this.currentMode === 'atomic') ? ATOMIC_PRESETS : COSMIC_PRESETS;
    Object.keys(presetMap).forEach((key) => {
      const p = presetMap[key];
      const btn = document.createElement('button');
      btn.className = `preset-btn ${key === this.currentPresetKey ? 'active' : ''}`;
      btn.id = `btn-preset-${key}`;
      btn.dataset.preset = key;

      let icon = '♾️';
      if (p.mode === 'atomic') {
        icon = (key === 'wannier') ? '⚡' : (key === 'frozen_planet' ? '🛡️' : (key === 'rydberg_2s2p' ? '🪐' : '📐'));
      } else {
        icon = (key === 'figure8') ? '♾️' : (key === 'lagrange' ? '📐' : (key === 'trisolaris' ? '☀️' : (key === 'pythagoras' ? '🔺' : '↔️')));
      }

      btn.innerHTML = `
        <span class="preset-icon">${icon}</span>
        <div class="preset-info">
          <span class="preset-name">${p.name}</span>
          <span class="preset-desc">${p.description}</span>
        </div>
      `;
      btn.addEventListener('click', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.loadPreset(key);
      });
      container.appendChild(btn);
    });
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
    const unit = (this.currentMode === 'atomic') ? 'a.u.' : 'AU';
    this.valA1.textContent = `${sides.a1.toFixed(2)} ${unit}`;
    this.valA2.textContent = `${sides.a2.toFixed(2)} ${unit}`;
    this.valA3.textContent = `${sides.a3.toFixed(2)} ${unit}`;

    const maxDist = Math.max(sides.a1, sides.a2, sides.a3, 4.0);
    this.barA1.style.width = `${Math.min(100, (sides.a1 / maxDist) * 100)}%`;
    this.barA2.style.width = `${Math.min(100, (sides.a2 / maxDist) * 100)}%`;
    this.barA3.style.width = `${Math.min(100, (sides.a3 / maxDist) * 100)}%`;

    // Fläche & Symmetrie
    this.valArea.textContent = `${area.toFixed(2)} ${unit}²`;
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
    const stability = this.physics.evaluateStability();
    if (this.frameCount % 4 === 0) {
      this.stabilityBadge.textContent = `${stability.score}% ${stability.state === 'PERIODISCH' ? 'STABIL' : stability.state}`;
      this.stabilityBadge.style.color = stability.color;
      this.stabilityBadge.style.borderColor = `${stability.color}66`;
      this.stabilityBadge.style.background = `${stability.color}1c`;
      this.stabilityStatusText.textContent = stability.label;
      this.stabilityStatusText.style.color = stability.color;
      this.stabilityDescText.textContent = stability.desc || '';
    }

    // Multi-Eigenzeit Daten
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
        if (this.valCyclesCount) {
          const cycText = formulaPred.isSlingshotLocked 
            ? `${formulaPred.cycles_count} (Riss!)` 
            : `${formulaPred.cycles_count} / ~${formulaPred.n_pred}`;
          this.valCyclesCount.textContent = cycText;
        }
        if (this.valPhaseIntegral) {
          this.valPhaseIntegral.textContent = `Σ = ${formulaPred.sigma_integral.toFixed(1)}`;
        }

        this.trackerStatus.textContent = formulaPred.isSlingshotLocked ? 'Hyperbolischer Flug:' : 'Zeit bis Riss:';
        this.trackerPct.textContent = `${pct}%`;
        this.trackerBarFill.style.width = `${pct}%`;
        this.trackerBarFill.style.background = formulaPred.isSlingshotLocked 
          ? 'linear-gradient(90deg, #ff0055, #ff5400)' 
          : 'linear-gradient(90deg, #ffb703, #ff0055)';

        if (this.plainNumbersRow) {
          const tOrbitStr = formulaPred.t_orbit.toFixed(1);
          const tKeplerStr = formulaPred.t_kepler.toFixed(1);
          const tTotStr = formulaPred.t_pred_symbolic.toFixed(1);
          this.plainNumbersRow.innerHTML = `
            <span class="font-mono text-cyan" style="font-weight: 700;">${tTotStr} s</span> = (<span class="font-mono text-green">${formulaPred.n_pred} Zyklen</span> − 1) × <span class="font-mono">${tOrbitStr} s</span> + <span class="font-mono text-gold">${tKeplerStr} s</span>
          `;
        }
      }

      // ======================================================================
      // 🌟 MISSION CONTROL HUD AKTUALISIEREN (Auf einen Blick)
      // ======================================================================
      if (this.hudStatusBadge) {
        if (stability.state === 'PERIODISCH') {
          this.hudStatusBadge.textContent = 'STABIL (Grenzzyklus)';
          this.hudStatusBadge.className = 'hud-status-badge status-stable';
          this.hudStatusSub.textContent = 'Zwei-Phasen: Harmonischer periodischer Orbit';
        } else if (stability.state === 'AUSWURF') {
          const ejectLabel = (this.currentMode === 'atomic') ? 'AUTOIONISIERT (Flucht)' : 'AUSWURF / ZERFALL';
          this.hudStatusBadge.textContent = ejectLabel;
          this.hudStatusBadge.className = 'hud-status-badge status-ionized';
          this.hudStatusSub.textContent = `${formulaPred.candidateName} hat Fluchtgeschwindigkeit erreicht`;
        } else {
          const isPhase1 = formulaPred.cycles_count <= 4;
          if (isPhase1) {
            this.hudStatusBadge.textContent = `PHASE 1 (N = ${formulaPred.cycles_count} ≤ 4)`;
            this.hudStatusBadge.className = 'hud-status-badge status-phase1';
            this.hudStatusSub.textContent = 'Deterministisch · Initialpaarung erhalten (>70%)';
          } else {
            this.hudStatusBadge.textContent = `PHASE 2 (N = ${formulaPred.cycles_count} ≥ 5)`;
            this.hudStatusBadge.className = 'hud-status-badge status-phase2';
            this.hudStatusSub.textContent = 'Ergodischer Poisson-Zerfall · Gedächtnis gelöscht';
          }
        }

        // Doppler Gauge
        const cosA = formulaPred.cos_alpha;
        const needlePct = Math.max(3, Math.min(97, (cosA + 1.0) * 50));
        this.hudDopplerNeedle.style.left = `${needlePct}%`;
        this.hudDopplerText.textContent = `${cosA >= 0 ? '+' : ''}${cosA.toFixed(2)}`;
        if (cosA < -0.3) {
          this.hudDopplerDesc.textContent = 'Tor offen · Frontaler Impulsübertrag';
        } else if (cosA > 0.3) {
          this.hudDopplerDesc.textContent = 'Tor verriegelt · Resonanz-Schutzschild';
        } else {
          this.hudDopplerDesc.textContent = 'Kozai-Neutral · Orthogonaler Orbit';
        }

        // Shear Gauge
        const shear = formulaPred.delta_phi_rel;
        const shearPct = Math.max(5, Math.min(100, shear * 100));
        this.hudShearFill.style.width = `${shearPct}%`;
        this.hudShearText.textContent = `ΔΦ = ${shear.toFixed(2)}`;
        this.hudShearDesc.textContent = (shear > 0.45) ? 'Starker Paarschutz (Tiefer Trichter)' : 'Symmetrisch (Geringer Schutz)';

        // Predictor
        if (stability.state === 'PERIODISCH') {
          this.hudPredTarget.textContent = 'Kein Zerfall';
          this.hudPredConf.textContent = '100%';
          this.hudPredDetail.textContent = 'Grenzzyklus im Formraum';
        } else {
          this.hudPredTarget.textContent = formulaPred.candidateName;
          this.hudPredConf.textContent = formulaPred.isSlingshotLocked ? '100%' : `${formulaPred.confidencePct}%`;
          this.hudPredDetail.textContent = `~${formulaPred.t_pred_symbolic.toFixed(1)}s · Zyklen: ~${formulaPred.n_pred}`;
        }
      }
    }

    // Oszilloskop Puffer aktualisieren
    if (this.frameCount % 2 === 0) {
      this.historyData.omega.push(omega);
      this.historyData.omega.shift();

      this.historyData.a1.push(sides.a1);
      this.historyData.a1.shift();

      this.historyData.a2.push(sides.a2);
      this.historyData.a2.shift();

      this.historyData.a3.push(sides.a3);
      this.historyData.a3.shift();

      if (this.chartMode === 'time') {
        this.drawOscilloscope();
      } else {
        this.drawPhaseSpace();
      }
    }
  }

  drawOscilloscope() {
    const ctx = this.oscCtx;
    const w = this.oscCanvas.width;
    const h = this.oscCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Raster
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.25); ctx.lineTo(w, h * 0.25);
    ctx.moveTo(0, h * 0.50); ctx.lineTo(w, h * 0.50);
    ctx.moveTo(0, h * 0.75); ctx.lineTo(w, h * 0.75);
    ctx.stroke();

    const drawLine = (data, color, maxVal, lineWidth = 1) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      const step = w / (data.length - 1);
      for (let i = 0; i < data.length; i++) {
        const val = Math.min(maxVal, Math.max(0, data[i]));
        const y = h - (val / maxVal) * (h * 0.85) - 4;
        const x = i * step;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    if (!this.smoothMaxA) this.smoothMaxA = 5.0;
    if (!this.smoothMaxO) this.smoothMaxO = 6.0;

    const targetMaxA = Math.max(5.0, ...this.historyData.a1, ...this.historyData.a2, ...this.historyData.a3);
    const targetMaxO = Math.max(6.0, ...this.historyData.omega);

    this.smoothMaxA += (targetMaxA - this.smoothMaxA) * 0.05;
    this.smoothMaxO += (targetMaxO - this.smoothMaxO) * 0.05;

    drawLine(this.historyData.a1, 'rgba(0, 240, 255, 0.65)', this.smoothMaxA, 1.2);
    drawLine(this.historyData.a2, 'rgba(255, 0, 127, 0.65)', this.smoothMaxA, 1.2);
    drawLine(this.historyData.a3, 'rgba(0, 255, 136, 0.65)', this.smoothMaxA, 1.2);
    drawLine(this.historyData.omega, '#ff9e00', this.smoothMaxO, 2.0);
  }

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

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 0); ctx.lineTo(w * 0.5, h);
    ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    let minO = Infinity, maxO = -Infinity;
    let maxRate = 1.0;
    for (const pt of history) {
      if (pt.omega < minO) minO = pt.omega;
      if (pt.omega > maxO) maxO = pt.omega;
      if (Math.abs(pt.dOmega) > maxRate) maxRate = Math.abs(pt.dOmega);
    }
    const rangeO = Math.max(0.5, maxO - minO);
    const midO = (minO + maxO) / 2.0;

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

  // ==========================================================================
  // ⚡ INTERAKTIVES MONTE-CARLO LABOR (Simulation mit X Systemen)
  // ==========================================================================
  initMonteCarloModal() {
    this.mcModal = document.getElementById('monte-carlo-modal');
    this.btnOpenMC = document.getElementById('btn-open-monte-carlo');
    this.btnCloseMC = document.getElementById('btn-close-monte-carlo');

    this.mcModeCosmic = document.getElementById('mc-mode-cosmic');
    this.mcModeAtomic = document.getElementById('mc-mode-atomic');
    this.mcBatchCount = 1000;
    this.mcBatchMode = 'cosmic';
    this.mcHasRun = false;

    this.btnRunBatch = document.getElementById('btn-run-batch');
    this.mcProgressBar = document.getElementById('mc-progress-bar');
    this.mcProgressText = document.getElementById('mc-progress-text');
    this.mcProgressWrap = document.getElementById('mc-progress-wrapper');

    this.mcKpiDecayLabel = document.getElementById('mc-kpi-decay-label');
    this.mcStatDecayPct = document.getElementById('mc-stat-decay-pct');
    this.mcStatDecays = document.getElementById('mc-stat-decays');
    this.mcStatPhaseRatio = document.getElementById('mc-stat-phase-ratio');
    this.mcStatPhases = document.getElementById('mc-stat-phases');
    this.mcStatLambda = document.getElementById('mc-stat-lambda');
    this.mcStatR2 = document.getElementById('mc-stat-r2');
    this.mcStatAccuracy = document.getElementById('mc-stat-accuracy');
    this.mcScatterCanvas = document.getElementById('mc-scatter-canvas');
    this.mcScatterCtx = this.mcScatterCanvas ? this.mcScatterCanvas.getContext('2d') : null;

    if (this.btnOpenMC) {
      this.btnOpenMC.addEventListener('click', () => {
        this.mcModal.classList.add('active');
        if (!this.mcHasRun) {
          this.executeMonteCarloBatch();
        }
      });
    }

    if (this.btnCloseMC) {
      this.btnCloseMC.addEventListener('click', () => {
        this.mcModal.classList.remove('active');
      });
    }

    this.mcModal.addEventListener('click', (e) => {
      if (e.target === this.mcModal) {
        this.mcModal.classList.remove('active');
      }
    });

    if (this.mcModeCosmic && this.mcModeAtomic) {
      this.mcModeCosmic.addEventListener('click', () => {
        this.mcBatchMode = 'cosmic';
        this.mcModeCosmic.classList.add('active');
        this.mcModeAtomic.classList.remove('active');
        this.mcKpiDecayLabel.textContent = 'Auswurf- / Zerfallsrate';
      });

      this.mcModeAtomic.addEventListener('click', () => {
        this.mcBatchMode = 'atomic';
        this.mcModeAtomic.classList.add('active');
        this.mcModeCosmic.classList.remove('active');
        this.mcKpiDecayLabel.textContent = 'Coulomb-Autoionisationsrate';
      });
    }

    const countBtns = document.querySelectorAll('.mc-count-btn');
    countBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        countBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.mcBatchCount = parseInt(btn.dataset.count, 10);
      });
    });

    if (this.btnRunBatch) {
      this.btnRunBatch.addEventListener('click', () => {
        this.executeMonteCarloBatch();
      });
    }
  }

  async executeMonteCarloBatch() {
    this.mcHasRun = true;
    if (this.btnRunBatch) this.btnRunBatch.disabled = true;
    if (this.mcProgressWrap) {
      this.mcProgressWrap.style.display = 'block';
      this.mcProgressBar.style.width = '0%';
      this.mcProgressText.textContent = `Berechne 0 / ${this.mcBatchCount.toLocaleString()} Systeme...`;
    }

    const result = await PhysicsEngine.runMonteCarloBatch(
      this.mcBatchCount,
      this.mcBatchMode,
      (current, total) => {
        const pct = Math.round((current / total) * 100);
        if (this.mcProgressBar) this.mcProgressBar.style.width = `${pct}%`;
        if (this.mcProgressText) this.mcProgressText.textContent = `Berechne ${current.toLocaleString()} / ${total.toLocaleString()} Systeme (${pct}%)...`;
      }
    );

    if (this.mcProgressWrap) this.mcProgressWrap.style.display = 'none';
    if (this.btnRunBatch) this.btnRunBatch.disabled = false;

    const s = result.summary;
    this.mcStatDecayPct.textContent = `${s.decayPct}%`;
    this.mcStatDecays.textContent = `${s.totalDecays.toLocaleString()} / ${s.totalCount.toLocaleString()} zerfallen`;
    this.mcStatPhaseRatio.textContent = `${s.phase1Pct}% : ${s.phase2Pct}%`;
    this.mcStatPhases.textContent = `Phase 1 (N≤4): ${s.phase1Count.toLocaleString()} | Phase 2 (N≥5): ${s.phase2Count.toLocaleString()}`;
    this.mcStatLambda.textContent = `λ = ${s.lambda}`;
    this.mcStatR2.textContent = `Poisson-Fit R² = ${s.r2}`;
    this.mcStatAccuracy.textContent = `${s.accuracyPct}%`;

    this.drawMonteCarloScatter(result.points, s);
  }

  drawMonteCarloScatter(points, summary) {
    if (!this.mcScatterCanvas || !this.mcScatterCtx) return;
    const canvas = this.mcScatterCanvas;
    const ctx = this.mcScatterCtx;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, w, h);

    // Raster & Skalenlinien
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    for (let t = 0; t <= 35; t += 7) {
      const y = h - 30 - (t / 35) * (h - 55);
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();

      ctx.fillStyle = '#5c6882';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText(`${t}s`, 42, y + 3);
    }

    for (let c = -1.0; c <= 1.0; c += 0.5) {
      const x = 50 + ((c + 1.0) / 2.0) * (w - 70);
      ctx.beginPath();
      ctx.moveTo(x, 25);
      ctx.lineTo(x, h - 30);
      ctx.stroke();

      ctx.fillStyle = '#5c6882';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(c.toFixed(1), x, h - 14);
    }

    ctx.fillStyle = '#8e9bb4';
    ctx.font = '11px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Doppler-Tor cos(α)   [ -1.0: Sturm / Retrograd  ←──  0.0  ──→  +1.0: Schild / Prograd ]', w / 2, h - 2);

    ctx.fillStyle = 'rgba(255, 0, 85, 0.85)';
    ctx.font = 'bold 11px Outfit, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('▼ Wannier-Sattelkamm (cos α → -1.0)', 55, 42);

    ctx.fillStyle = 'rgba(0, 255, 136, 0.85)';
    ctx.textAlign = 'right';
    ctx.fillText('Frozen-Planet-Oase (cos α → +1.0) ▼', w - 25, 42);

    for (const p of points) {
      const x = 50 + ((p.cosAlpha + 1.0) / 2.0) * (w - 70);
      const y = h - 30 - (Math.min(35, p.tEject) / 35) * (h - 55);

      ctx.beginPath();
      ctx.arc(x, y, p.phase === 1 ? 2.8 : 2.0, 0, Math.PI * 2);
      ctx.fillStyle = (p.phase === 1) ? 'rgba(0, 240, 255, 0.8)' : 'rgba(255, 0, 85, 0.7)';
      ctx.fill();
    }
  }

  startLoop() {
    const loop = (currentTime) => {
      requestAnimationFrame(loop);

      if (!this.lastFrameTime) {
        this.lastFrameTime = currentTime;
      }
      const delta = Math.min((currentTime - this.lastFrameTime) / 1000.0, 0.05);
      this.lastFrameTime = currentTime;
      this.frameCount++;

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

      // DOM-Telemetrie & Oszilloskop auf 15 Hz drosseln für ultra-flüssige 60-144 FPS
      if (this.frameCount % 4 === 0) {
        this.updateTelemetry();
      }
    };

    requestAnimationFrame(loop);
  }
}

// Start beim Laden des Fensters
window.addEventListener('DOMContentLoaded', () => {
  new TrinityApp();
});
