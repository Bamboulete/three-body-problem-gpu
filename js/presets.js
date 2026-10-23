/**
 * Trinitäts-Topologie: Forschungs-Presets
 * Präzise Initialbedingungen für fundamentale 3-Körper-Zustände.
 */

import { Body } from './physics.js';

export const PRESETS = {
  figure8: {
    id: 'figure8',
    name: 'Figure-8 Choreographie',
    author: 'Chenciner & Montgomery (2000)',
    description: 'Drei identische Massen bewegen sich im stabilen Gleichgewicht hintereinander auf einer Schleife (Achterbahn-Orbit).',
    softening: 0.005,
    cameraDist: 4.5,
    bodies: () => {
      const x1 = -0.97000436;
      const y1 = 0.24308753;
      const vx1 = 0.46620531;
      const vy1 = 0.43236573;

      const vx3 = -2 * vx1;
      const vy3 = -2 * vy1;

      return [
        new Body('a', 'Körper A (Cyan)', 1.0, '#00f0ff', { x: x1, y: y1, z: 0 }, { vx: vx1, vy: vy1, vz: 0 }),
        new Body('b', 'Körper B (Gold)', 1.0, '#ffb703', { x: -x1, y: -y1, z: 0 }, { vx: vx1, vy: vy1, vz: 0 }),
        new Body('c', 'Körper C (Magenta)', 1.0, '#ff007f', { x: 0, y: 0, z: 0 }, { vx: vx3, vy: vy3, vz: 0 })
      ];
    }
  },

  lagrange: {
    id: 'lagrange',
    name: 'Lagrange Gleichseitig (L4/L5)',
    author: 'Joseph-Louis Lagrange (1772)',
    description: 'Drei Körper bilden ein rotierendes, gleichseitiges Dreieck. Die Geometrie GS bleibt starr, während das System rotiert.',
    softening: 0.02,
    cameraDist: 6.0,
    bodies: () => {
      const R = 2.2;
      const m = 1.0;
      // Winkelgeschwindigkeit omega = sqrt(G * (m1+m2+m3) / R^3)
      const omega = Math.sqrt((3.0 * 1.0) / Math.pow(R, 3)) * 0.76;

      const angles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
      const colors = ['#00f0ff', '#ffb703', '#ff007f'];
      const names = ['Körper A (Cyan)', 'Körper B (Gold)', 'Körper C (Magenta)'];

      return angles.map((ang, i) => {
        const x = R * Math.cos(ang);
        const y = R * Math.sin(ang);
        const vx = -omega * y;
        const vy = omega * x;
        return new Body(
          i === 0 ? 'a' : i === 1 ? 'b' : 'c',
          names[i],
          m,
          colors[i],
          { x, y, z: 0 },
          { vx, vy, vz: 0 }
        );
      });
    }
  },

  trisolaris: {
    id: 'trisolaris',
    name: 'Trisolaris Chaos',
    author: 'Cixin Liu / Klassische Chaotik',
    description: 'Drei Sterne mit unterschiedlichen Massen wirbeln in unregelmäßigen Schlingen umeinander. Hohe Deformationsdynamik im Ω-Feld.',
    softening: 0.05,
    cameraDist: 7.0,
    bodies: () => {
      return [
        new Body('a', 'Körper A (Cyan)', 1.4, '#00f0ff', { x: -1.8, y: 0.5, z: 0.2 }, { vx: 0.1, vy: -0.65, vz: 0.1 }),
        new Body('b', 'Körper B (Gold)', 1.1, '#ffb703', { x: 1.5, y: -0.8, z: -0.3 }, { vx: -0.4, vy: 0.5, vz: -0.1 }),
        new Body('c', 'Körper C (Magenta)', 0.9, '#ff007f', { x: 0.2, y: 1.6, z: 0.1 }, { vx: 0.35, vy: -0.15, vz: 0.05 })
      ];
    }
  },

  pythagoras: {
    id: 'pythagoras',
    name: 'Pythagoras Problem (Burrau 1913)',
    author: 'Carl Burrau (1913)',
    description: 'Massen 3, 4, 5 an den Ecken eines 3-4-5 rechtwinkligen Dreiecks, anfangs in Ruhe. Führt zu dramatischen Beinahe-Stößen und späterem Auswurf.',
    softening: 0.03,
    cameraDist: 9.0,
    bodies: () => {
      // Schwerpunkt zentriert
      return [
        new Body('a', 'Körper A (m=3)', 3.0, '#00f0ff', { x: 1.0, y: 3.0, z: 0 }, { vx: 0, vy: 0, vz: 0 }),
        new Body('b', 'Körper B (m=4)', 4.0, '#ffb703', { x: -2.0, y: -1.0, z: 0 }, { vx: 0, vy: 0, vz: 0 }),
        new Body('c', 'Körper C (m=5)', 5.0, '#ff007f', { x: 1.0, y: -1.0, z: 0 }, { vx: 0, vy: 0, vz: 0 })
      ];
    }
  },

  euler: {
    id: 'euler',
    name: 'Euler Kollinear',
    author: 'Leonhard Euler (1767)',
    description: 'Drei Körper auf einer geraden Linie. Symmetrischer instabiler Sattelpunkt – die kleinste Störung bricht die Kollinearität.',
    softening: 0.02,
    cameraDist: 6.5,
    bodies: () => {
      return [
        new Body('a', 'Körper A (Cyan)', 1.0, '#00f0ff', { x: -2.2, y: 0, z: 0 }, { vx: 0, vy: -0.7, vz: 0 }),
        new Body('b', 'Körper B (Gold)', 1.5, '#ffb703', { x: 0, y: 0, z: 0 }, { vx: 0, vy: 0.0, vz: 0 }),
        new Body('c', 'Körper C (Magenta)', 1.0, '#ff007f', { x: 2.2, y: 0, z: 0 }, { vx: 0, vy: 0.7, vz: 0 })
      ];
    }
  }
};
