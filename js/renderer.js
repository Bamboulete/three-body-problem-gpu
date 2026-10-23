/**
 * Trinitäts-Topologie: 3D-Renderer & Visualisierung
 * WebGL / Three.js Szene mit dynamischem Spannungs-Gewebe GS,
 * leuchtenden Sternenkörpern, Orbitalspuren und Formraum-Kamera.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// Realistische GLSL Solar-Shader (Oberflächen-Granulation & Korona-Strahlen)
// ============================================================================

const NOISE_GLSL = `
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

float fbm(vec3 p) {
  return 0.65 * snoise(p) + 0.35 * snoise(p * 2.05);
}
`;

const STAR_SURFACE_VERT = `
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const STAR_SURFACE_FRAG = `
uniform float uTime;
uniform vec3 uColorCore;
uniform vec3 uColorMid;
uniform vec3 uColorEdge;
uniform float uSpeed;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

${NOISE_GLSL}

void main() {
  vec3 norm = normalize(vNormal);
  vec3 viewDir = normalize(vViewPosition);

  // Brodelnde Konvektions-Granulation
  float t = uTime * uSpeed;
  vec3 noiseCoord = vPosition * 4.8 + vec3(sin(t * 0.4) * 0.6, t * 0.25, cos(t * 0.3) * 0.6);
  float n = fbm(noiseCoord);

  // Randverdunkelung / Fresnel Glühen
  float fresnel = 1.0 - max(0.0, dot(norm, viewDir));

  // Farbmischung von dunklerem Plasma zu glühenden Filamenten
  vec3 col = mix(uColorEdge, uColorMid, clamp(n * 0.6 + 0.5, 0.0, 1.0));
  col = mix(col, uColorCore, pow(clamp(n, 0.0, 1.0), 2.2));
  // Glühender Rand
  col += uColorCore * pow(fresnel, 2.2) * 1.9;

  gl_FragColor = vec4(col, 1.0);
}
`;

const CORONA_VERT = `
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const CORONA_FRAG = `
uniform float uTime;
uniform vec3 uColorCorona;
uniform vec3 uColorCore;
uniform float uSpeed;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

${NOISE_GLSL}

void main() {
  vec3 norm = normalize(vNormal);
  vec3 viewDir = normalize(vViewPosition);

  // Ethereales, samtweiches Randglühen (Fresnel)
  float fresnel = 1.0 - abs(dot(norm, viewDir));
  float rim = pow(fresnel, 3.2);

  // Strahlende, organisch pulsierende Koronar-Eruptionen (optimierter schneller Shader)
  float t = uTime * uSpeed;
  vec3 rayCoord = normalize(vPosition) * 3.8 + vec3(sin(t * 0.3) * 0.3, t * 0.35, cos(t * 0.2) * 0.3);
  float rays = snoise(rayCoord) * 0.5 + 0.5;

  float flare = rim * (0.3 + 0.7 * rays);
  vec3 coronaCol = mix(uColorCorona, uColorCore, pow(fresnel, 2.5));

  // Sanft auslaufende Transparenz am Rand
  gl_FragColor = vec4(coronaCol, flare * 0.95);
}
`;

export class SceneRenderer {
  constructor(containerElement) {
    this.container = containerElement;
    this.viewMode = 'lab'; // 'lab' oder 'shape'
    this.showTrails = true;
    this.showTriangleMesh = true;
    this.showVectors = true;
    this.mode = 'cosmic'; // 'cosmic' oder 'atomic'

    this.trailHistoryLength = 160;
    this.trailsData = [[], [], []];

    this.initScene();
    this.initObjects();
    this.initAtomicObjects();
    this.initStarfield();

    window.addEventListener('resize', () => this.onWindowResize());
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x07090e, 0.015);

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 1000);
    this.camera.position.set(0, 4.5, 9);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    this.renderer.setClearColor(0x07090e, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxDistance = 50;
    this.controls.minDistance = 1.5;

    // Sanfte Umgebungsbeleuchtung
    const ambientLight = new THREE.AmbientLight(0x161c28, 1.2);
    this.scene.add(ambientLight);
  }

  initObjects() {
    // 3 Realistische Sonnen (Klasse O Blau, Klasse G Gelb, Klasse M Rot)
    this.bodyObjects = [];

    const starConfigs = [
      {
        name: 'Stern A (Typ O - Blauer Hyperriese)',
        baseRadius: 0.16,
        coronaRadius: 0.38,
        coreColor: 0xffffff,
        midColor: 0x00f0ff,
        edgeColor: 0x0044ff,
        coronaColor: 0x00b4d8,
        lightColor: 0x00e5ff,
        speed: 1.2
      },
      {
        name: 'Stern B (Typ G - Gelbe Sonne)',
        baseRadius: 0.15,
        coronaRadius: 0.36,
        coreColor: 0xfffcf0,
        midColor: 0xffb703,
        edgeColor: 0xd00000,
        coronaColor: 0xff6a00,
        lightColor: 0xffaa00,
        speed: 0.95
      },
      {
        name: 'Stern C (Typ M - Roter Zwerg/Riese)',
        baseRadius: 0.14,
        coronaRadius: 0.34,
        coreColor: 0xffe8e8,
        midColor: 0xe63946,
        edgeColor: 0x4a0404,
        coronaColor: 0xd90429,
        lightColor: 0xff1e56,
        speed: 0.75
      }
    ];

    const sphereGeo = new THREE.SphereGeometry(1.0, 48, 48);
    const coronaGeo = new THREE.SphereGeometry(1.0, 32, 32);

    for (let i = 0; i < 3; i++) {
      const cfg = starConfigs[i];
      const group = new THREE.Group();

      // 1. Animierte brodelnde Plasma-Oberfläche (Custom Shader)
      const surfaceMat = new THREE.ShaderMaterial({
        vertexShader: STAR_SURFACE_VERT,
        fragmentShader: STAR_SURFACE_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uColorCore: { value: new THREE.Color(cfg.coreColor) },
          uColorMid: { value: new THREE.Color(cfg.midColor) },
          uColorEdge: { value: new THREE.Color(cfg.edgeColor) },
          uSpeed: { value: cfg.speed }
        }
      });
      const core = new THREE.Mesh(sphereGeo, surfaceMat);
      core.scale.setScalar(cfg.baseRadius);
      group.add(core);

      // 2. Volumetrische, pulsierende Korona (Atmosphäre & Strahlen)
      const coronaMat = new THREE.ShaderMaterial({
        vertexShader: CORONA_VERT,
        fragmentShader: CORONA_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uColorCore: { value: new THREE.Color(cfg.coreColor) },
          uColorCorona: { value: new THREE.Color(cfg.coronaColor) },
          uSpeed: { value: cfg.speed }
        },
        transparent: true,
        side: THREE.FrontSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const corona = new THREE.Mesh(coronaGeo, coronaMat);
      corona.scale.setScalar(cfg.coronaRadius);
      group.add(corona);

      // 3. Protuberanzen-Ring (Solar Flare Loops)
      const ringGeo = new THREE.RingGeometry(cfg.baseRadius * 1.05, cfg.baseRadius * 1.45, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: cfg.coronaColor,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending
      });
      const flareRing = new THREE.Mesh(ringGeo, ringMat);
      flareRing.rotation.x = Math.PI * 0.35;
      flareRing.rotation.y = Math.PI * 0.2;
      group.add(flareRing);

      // 4. Dynamisches Punktlicht für Umgebungsbeleuchtung
      const light = new THREE.PointLight(cfg.lightColor, 2.8, 14, 1.1);
      group.add(light);

      this.scene.add(group);
      this.bodyObjects.push({
        group,
        core,
        corona,
        flareRing,
        surfaceMat,
        coronaMat,
        light,
        cfg
      });
    }

    // Trinitäts-Dreieck (GS) Spannungs-Gewebe Mesh
    const triGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(9); // 3 Vertices * (x,y,z)
    triGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    this.triangleMaterial = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.triangleMesh = new THREE.Mesh(triGeometry, this.triangleMaterial);
    this.scene.add(this.triangleMesh);

    // Kanten des Dreiecks (GS)
    const edgeGeo = new THREE.BufferGeometry();
    const edgePositions = new Float32Array(12); // A -> B -> C -> A
    edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3));

    this.edgeMaterial = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending
    });
    this.edgeLines = new THREE.Line(edgeGeo, this.edgeMaterial);
    this.scene.add(this.edgeLines);

    // Orbitalspuren (Trails) für jeden Körper
    this.trailLines = [];
    for (let i = 0; i < 3; i++) {
      const trailGeo = new THREE.BufferGeometry();
      const trailPos = new Float32Array(this.trailHistoryLength * 3);
      trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));

      const trailMat = new THREE.LineBasicMaterial({
        color: starConfigs[i].lightColor,
        transparent: true,
        opacity: 0.65,
        blending: THREE.AdditiveBlending
      });

      const trailLine = new THREE.Line(trailGeo, trailMat);
      this.scene.add(trailLine);
      this.trailLines.push({ line: trailLine, geometry: trailGeo });
    }

    // Schwerpunkt-Reticle (Barycenter Anchor)
    const baryGeo = new THREE.RingGeometry(0.08, 0.11, 24);
    const baryMat = new THREE.MeshBasicMaterial({
      color: 0x5c6882,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.5
    });
    this.barycenterMesh = new THREE.Mesh(baryGeo, baryMat);
    this.barycenterMesh.rotation.x = Math.PI / 2;
    this.scene.add(this.barycenterMesh);
  }

  initStarfield() {
    const starCount = 1800;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      const radius = 60 + Math.random() * 80;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);

      positions[i] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i + 2] = radius * Math.cos(phi);

      const tint = 0.7 + Math.random() * 0.3;
      colors[i] = tint * 0.8;
      colors[i + 1] = tint * 0.9;
      colors[i + 2] = tint;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.35,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      depthWrite: false
    });

    this.starfield = new THREE.Points(geometry, material);
    this.scene.add(this.starfield);
  }

  // Trails leeren beim Wechsel von Presets
  clearTrails() {
    this.trailsData = [[], [], []];
    for (let i = 0; i < 3; i++) {
      const posAttr = this.trailLines[i].geometry.attributes.position;
      posAttr.array.fill(0);
      posAttr.needsUpdate = true;
      this.trailLines[i].geometry.setDrawRange(0, 0);
    }
  }

  initAtomicObjects() {
    this.atomicBodyObjects = [];

    // ========================================================================
    // Körper 0: HELIUM-KERN He²⁺ (Alpha-Teilchen = 2 Protonen + 2 Neutronen)
    // ========================================================================
    const nucGroup = new THREE.Group();
    const nucleonsGroup = new THREE.Group();

    // 2 Protonen (Gold-Rot leuchtend mit positiver Kernladung)
    const protonMat = new THREE.MeshStandardMaterial({
      color: 0xff3a00,
      emissive: 0xff2200,
      emissiveIntensity: 0.9,
      roughness: 0.2,
      metalness: 0.2
    });
    // 2 Neutronen (Neutrales Hochenergie-Silber/Eisblau)
    const neutronMat = new THREE.MeshStandardMaterial({
      color: 0xa0d8ef,
      emissive: 0x1b4965,
      emissiveIntensity: 0.7,
      roughness: 0.2,
      metalness: 0.3
    });

    const nucleonGeo = new THREE.SphereGeometry(0.065, 20, 20);
    const d = 0.045; // Nukleonen-Abstand im Kern

    const p1 = new THREE.Mesh(nucleonGeo, protonMat);
    p1.position.set(d, d, d);
    nucleonsGroup.add(p1);

    const p2 = new THREE.Mesh(nucleonGeo, protonMat);
    p2.position.set(-d, -d, d);
    nucleonsGroup.add(p2);

    const n1 = new THREE.Mesh(nucleonGeo, neutronMat);
    n1.position.set(-d, d, -d);
    nucleonsGroup.add(n1);

    const n2 = new THREE.Mesh(nucleonGeo, neutronMat);
    n2.position.set(d, -d, -d);
    nucleonsGroup.add(n2);

    nucGroup.add(nucleonsGroup);

    // Starke Kernkraft-Hülle (Shimmering Strong Force Shell)
    const nucShellGeo = new THREE.SphereGeometry(0.14, 24, 24);
    const nucShellMat = new THREE.MeshBasicMaterial({
      color: 0xffb703,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const nucShell = new THREE.Mesh(nucShellGeo, nucShellMat);
    nucGroup.add(nucShell);

    const nucLight = new THREE.PointLight(0xffb703, 2.2, 12, 1.2);
    nucGroup.add(nucLight);

    nucGroup.visible = false;
    this.scene.add(nucGroup);

    this.atomicBodyObjects.push({
      group: nucGroup,
      nucleonsGroup,
      shell: nucShell,
      light: nucLight,
      type: 'nucleus'
    });

    // ========================================================================
    // Körper 1: ELEKTRON e₁ (Subatomarer Quanten-Lichtpunkt + de-Broglie-Wellen)
    // ========================================================================
    const e1Group = new THREE.Group();
    const e1CoreGeo = new THREE.SphereGeometry(0.035, 16, 16);
    const e1CoreMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const e1Core = new THREE.Mesh(e1CoreGeo, e1CoreMat);
    e1Group.add(e1Core);

    // de-Broglie Wellenringe (Cyan)
    const ringGeo1 = new THREE.RingGeometry(0.065, 0.082, 32);
    const ringMat1 = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.65,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    const e1Ring1 = new THREE.Mesh(ringGeo1, ringMat1);
    e1Ring1.rotation.x = Math.PI * 0.4;
    e1Group.add(e1Ring1);

    const ringGeo2 = new THREE.RingGeometry(0.095, 0.112, 32);
    const ringMat2 = new THREE.MeshBasicMaterial({
      color: 0x00b4d8,
      transparent: true,
      opacity: 0.45,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    const e1Ring2 = new THREE.Mesh(ringGeo2, ringMat2);
    e1Ring2.rotation.y = Math.PI * 0.45;
    e1Group.add(e1Ring2);

    const e1HaloGeo = new THREE.SphereGeometry(0.09, 16, 16);
    const e1HaloMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.28,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const e1Halo = new THREE.Mesh(e1HaloGeo, e1HaloMat);
    e1Group.add(e1Halo);

    const e1Light = new THREE.PointLight(0x00f0ff, 1.8, 10, 1.2);
    e1Group.add(e1Light);

    e1Group.visible = false;
    this.scene.add(e1Group);

    this.atomicBodyObjects.push({
      group: e1Group,
      core: e1Core,
      ring1: e1Ring1,
      ring2: e1Ring2,
      light: e1Light,
      type: 'electron'
    });

    // ========================================================================
    // Körper 2: ELEKTRON e₂ (Subatomarer Quanten-Lichtpunkt + de-Broglie-Wellen)
    // ========================================================================
    const e2Group = new THREE.Group();
    const e2CoreGeo = new THREE.SphereGeometry(0.035, 16, 16);
    const e2CoreMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const e2Core = new THREE.Mesh(e2CoreGeo, e2CoreMat);
    e2Group.add(e2Core);

    // de-Broglie Wellenringe (Magenta)
    const ringGeo3 = new THREE.RingGeometry(0.065, 0.082, 32);
    const ringMat3 = new THREE.MeshBasicMaterial({
      color: 0xff007f,
      transparent: true,
      opacity: 0.65,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    const e2Ring1 = new THREE.Mesh(ringGeo3, ringMat3);
    e2Ring1.rotation.x = Math.PI * 0.35;
    e2Group.add(e2Ring1);

    const ringGeo4 = new THREE.RingGeometry(0.095, 0.112, 32);
    const ringMat4 = new THREE.MeshBasicMaterial({
      color: 0x9d4edd,
      transparent: true,
      opacity: 0.45,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    const e2Ring2 = new THREE.Mesh(ringGeo4, ringMat4);
    e2Ring2.rotation.y = Math.PI * 0.35;
    e2Group.add(e2Ring2);

    const e2HaloGeo = new THREE.SphereGeometry(0.09, 16, 16);
    const e2HaloMat = new THREE.MeshBasicMaterial({
      color: 0xff007f,
      transparent: true,
      opacity: 0.28,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const e2Halo = new THREE.Mesh(e2HaloGeo, e2HaloMat);
    e2Group.add(e2Halo);

    const e2Light = new THREE.PointLight(0xff007f, 1.8, 10, 1.2);
    e2Group.add(e2Light);

    e2Group.visible = false;
    this.scene.add(e2Group);

    this.atomicBodyObjects.push({
      group: e2Group,
      core: e2Core,
      ring1: e2Ring1,
      ring2: e2Ring2,
      light: e2Light,
      type: 'electron'
    });
  }

  // Modus umschalten (Kosmos vs. Quanten-Helium-Atom)
  setMode(mode) {
    this.mode = mode;
    const isAtomic = (mode === 'atomic');

    // Kosmische Sonnen umschalten
    for (let i = 0; i < this.bodyObjects.length; i++) {
      this.bodyObjects[i].group.visible = !isAtomic;
    }

    // Atomkern & Elektronen umschalten
    for (let i = 0; i < this.atomicBodyObjects.length; i++) {
      this.atomicBodyObjects[i].group.visible = isAtomic;
    }

    if (isAtomic) {
      if (this.trailLines && this.trailLines.length >= 3) {
        this.trailLines[0].line.material.color.setHex(0xffb703); // Alpha-Kern Spur
        this.trailLines[1].line.material.color.setHex(0x00f0ff); // e1 Cyan Spur
        this.trailLines[2].line.material.color.setHex(0xff007f); // e2 Magenta Spur
      }
      this.triangleMaterial.opacity = 0.16;
      this.edgeMaterial.opacity = 0.55;
    } else {
      if (this.trailLines && this.trailLines.length >= 3) {
        this.trailLines[0].line.material.color.setHex(0x00e5ff);
        this.trailLines[1].line.material.color.setHex(0xffaa00);
        this.trailLines[2].line.material.color.setHex(0xff1e56);
      }
      this.triangleMaterial.opacity = 0.22;
      this.edgeMaterial.opacity = 0.75;
    }
  }

  // Update der 3D-Darstellung pro Animationsframe
  update(physics, frameCount) {
    if (physics.bodies.length < 3) return;

    const [bA, bB, bC] = physics.bodies;
    const pA = bA.pos;
    const pB = bB.pos;
    const pC = bC.pos;

    const stars = [pA, pB, pC];
    const timeSec = performance.now() * 0.001;

    if (this.mode === 'atomic') {
      // ⚛️ QUANTEN-HELIUM UPDATE:
      for (let i = 0; i < 3; i++) {
        const obj = this.atomicBodyObjects[i];
        obj.group.position.set(stars[i].x, stars[i].y, stars[i].z);
      }

      // Helium-Kern He²⁺: Taumelnde 4-Nukleonen-Rotation & pulsierende Hülle
      const nuc = this.atomicBodyObjects[0];
      nuc.nucleonsGroup.rotation.x += 0.016;
      nuc.nucleonsGroup.rotation.y += 0.022;
      nuc.shell.scale.setScalar(1.0 + Math.sin(timeSec * 3.5) * 0.06);

      // Elektronen de-Broglie Wellenringe rotieren
      const e1 = this.atomicBodyObjects[1];
      e1.ring1.rotation.z += 0.04;
      e1.ring2.rotation.x += 0.03;

      const e2 = this.atomicBodyObjects[2];
      e2.ring1.rotation.z -= 0.04;
      e2.ring2.rotation.y += 0.03;
    } else {
      // 🌌 KOSMISCHE STERNE UPDATE:
      for (let i = 0; i < 3; i++) {
        const body = physics.bodies[i];
        const obj = this.bodyObjects[i];
        obj.group.position.set(stars[i].x, stars[i].y, stars[i].z);

        const scale = Math.cbrt(body.mass);
        obj.core.scale.setScalar(obj.cfg.baseRadius * scale);
        obj.corona.scale.setScalar(obj.cfg.coronaRadius * scale);
        obj.flareRing.scale.setScalar(scale);

        if (obj.surfaceMat && obj.surfaceMat.uniforms.uTime) {
          obj.surfaceMat.uniforms.uTime.value = timeSec;
        }
        if (obj.coronaMat && obj.coronaMat.uniforms.uTime) {
          obj.coronaMat.uniforms.uTime.value = timeSec;
        }
        obj.flareRing.rotation.z += 0.007 * (i === 1 ? -1 : 1);
        obj.light.intensity = 2.6 + Math.sin(timeSec * 2.2 + i * 1.5) * 0.35;
      }
    }

    // 2. Trinitäts-Dreieck (GS) Spannungs-Gewebe Geometrie updaten
    if (this.showTriangleMesh || this.showVectors) {
      const triPos = this.triangleMesh.geometry.attributes.position.array;
      triPos[0] = pA.x; triPos[1] = pA.y; triPos[2] = pA.z;
      triPos[3] = pB.x; triPos[4] = pB.y; triPos[5] = pB.z;
      triPos[6] = pC.x; triPos[7] = pC.y; triPos[8] = pC.z;
      this.triangleMesh.geometry.attributes.position.needsUpdate = true;

      // Kanten A -> B -> C -> A
      const edgePos = this.edgeLines.geometry.attributes.position.array;
      edgePos[0] = pA.x;  edgePos[1] = pA.y;  edgePos[2] = pA.z;
      edgePos[3] = pB.x;  edgePos[4] = pB.y;  edgePos[5] = pB.z;
      edgePos[6] = pC.x;  edgePos[7] = pC.y;  edgePos[8] = pC.z;
      edgePos[9] = pA.x;  edgePos[10] = pA.y; edgePos[11] = pA.z;
      this.edgeLines.geometry.attributes.position.needsUpdate = true;

      // Dynamische Farbänderung des GS-Gewebes basierend auf Ω-Spannung!
      const omega = physics.calculateOmega();
      const tensionFactor = Math.min(1.0, Math.max(0.0, (omega - 0.8) / 3.5));

      const tensionColor = new THREE.Color();
      if (tensionFactor < 0.5) {
        tensionColor.lerpColors(new THREE.Color(0x00f0ff), new THREE.Color(0xffb703), tensionFactor * 2);
      } else {
        tensionColor.lerpColors(new THREE.Color(0xffb703), new THREE.Color(0xff0055), (tensionFactor - 0.5) * 2);
      }

      this.triangleMaterial.color = tensionColor;
      this.triangleMaterial.opacity = 0.15 + tensionFactor * 0.45;
      this.edgeMaterial.color = tensionColor;
      this.edgeMaterial.opacity = 0.55 + tensionFactor * 0.4;
    }

    this.triangleMesh.visible = this.showTriangleMesh;
    this.edgeLines.visible = this.showVectors;

    // 3. Orbitalspuren (Trails) historisieren
    if (this.showTrails && frameCount % 2 === 0) {
      for (let i = 0; i < 3; i++) {
        const hist = this.trailsData[i];
        hist.unshift({ x: stars[i].x, y: stars[i].y, z: stars[i].z });
        if (hist.length > this.trailHistoryLength) {
          hist.pop();
        }

        const posAttr = this.trailLines[i].geometry.attributes.position;
        const arr = posAttr.array;
        for (let j = 0; j < hist.length; j++) {
          arr[j * 3] = hist[j].x;
          arr[j * 3 + 1] = hist[j].y;
          arr[j * 3 + 2] = hist[j].z;
        }
        posAttr.needsUpdate = true;
        this.trailLines[i].geometry.setDrawRange(0, hist.length);
        this.trailLines[i].line.visible = true;
      }
    } else if (!this.showTrails) {
      for (let i = 0; i < 3; i++) {
        this.trailLines[i].line.visible = false;
      }
    }

    // 4. Schwerpunkt (Barycenter) positionieren
    const cm = physics.getCenterOfMass();
    this.barycenterMesh.position.set(cm.x, cm.y, cm.z);

    // 5. Kameraführung je nach Blickwinkel-Modus:
    if (this.viewMode === 'shape') {
      // TRINITÄTS-FORMRAUM:
      // Verhindere extremes Eintauchen ("riesen groß") beim Passieren des Zentrums/Kollinearität!
      if (!this.stableNormal) {
        this.stableNormal = new THREE.Vector3(0, 0, 1);
      }

      const u = new THREE.Vector3(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
      const v = new THREE.Vector3(pC.x - pA.x, pC.y - pA.y, pC.z - pA.z);
      const currentCross = new THREE.Vector3().crossVectors(u, v);
      const crossLen = currentCross.length();

      // Nur aktualisieren, wenn das Dreieck nicht zu einer Linie kollabiert (crossLen > 0.15)
      if (crossLen > 0.15) {
        currentCross.divideScalar(crossLen);
        // Vorzeichen-Kontinuität: Verhindere 180-Grad-Flip durch die Bildebene
        if (currentCross.dot(this.stableNormal) < 0) {
          currentCross.negate();
        }
        this.stableNormal.lerp(currentCross, 0.05);
      }

      // Fester stabiler Betrachtungsabstand (kein Durchtauchen durch die Körper!)
      const targetPos = new THREE.Vector3(cm.x, cm.y, cm.z).addScaledVector(this.stableNormal, 7.5);
      this.camera.position.lerp(targetPos, 0.06);
      this.controls.target.lerp(new THREE.Vector3(cm.x, cm.y, cm.z), 0.08);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  setViewMode(mode) {
    this.viewMode = mode;
    if (mode === 'lab') {
      this.controls.enableRotate = true;
    }
  }

  setCameraDistance(dist) {
    this.camera.position.set(0, dist * 0.45, dist);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  onWindowResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}
