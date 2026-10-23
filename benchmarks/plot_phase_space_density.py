"""
Generiere die hochauflösende 2D-Phasenraum-Karte (519.810.403 Systeme)
aus der 1-Milliarde-Simulation (scratch/phase_space_density_1b.npy).
"""

import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def plot_heatmap():
    print("Lade 500x500 Phasenraum-Dichtematrix (519 Millionen Systeme)...")
    grid = np.load("scratch/phase_space_density_1b.npy")
    
    # Grid hat 500 x 500 Bins:
    # X: Delta_Phi_rel in [0.0, 2.5]
    # Y: eta_rot in [0.0, 2.5]
    grid_masked = np.where(grid == 0, np.nan, grid)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    fig.patch.set_facecolor('#060913')
    ax.set_facecolor('#060913')
    
    extent = [0.0, 2.5, 0.0, 2.5] # [x_min, x_max, y_min, y_max]
    
    # Logarithmische Dichte mit inferno Colormap
    norm = mcolors.LogNorm(vmin=1, vmax=np.nanmax(grid_masked))
    im = ax.imshow(grid_masked, origin='lower', extent=extent, cmap='inferno', norm=norm, aspect='auto')
    
    # Farbskala
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Dichte der zerfallenen Systeme (519.810.403 Zerfälle)', color='#cbd5e1', fontsize=12, labelpad=12)
    cbar.ax.tick_params(colors='#94a3b8')
    
    # Reale astronomische Systeme einzeichnen
    real_systems = [
        {"name": "Orion Slingshot\n(AE Aurigae Auswurf)", "x": 0.705, "y": 0.005, "color": "#ff0055", "marker": "X", "size": 180},
        {"name": "Alpha Centauri AB + Proxima\n(Ewig stabil, >5 Mrd. J.)", "x": 1.636, "y": 0.007, "color": "#00ff88", "marker": "o", "size": 150},
        {"name": "Algol (Dämonenstern)\n(Hierarchisch stabil)", "x": 2.318, "y": 0.311, "color": "#00f0ff", "marker": "s", "size": 140},
        {"name": "GW Orionis\n(Verzerrte Scheibe)", "x": 1.264, "y": 0.824, "color": "#ffb703", "marker": "D", "size": 130},
        {"name": "KOI-126\n(Kepler Triple, η > 2.5)", "x": 1.304, "y": 2.40, "color": "#b5179e", "marker": "^", "size": 160},
    ]
    
    for s in real_systems:
        ax.scatter(s["x"], s["y"], color=s["color"], marker=s["marker"], s=s["size"], edgecolors='white', linewidth=1.5, zorder=5)
        offset_y = 0.08 if s["y"] < 1.8 else -0.15
        offset_x = 0.02 if s["x"] < 1.5 else -0.45
        ax.annotate(
            s["name"],
            xy=(s["x"], s["y"]),
            xytext=(s["x"] + offset_x, s["y"] + offset_y),
            color=s["color"],
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#0b1120", ec=s["color"], lw=1.2, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=s["color"], lw=1.2),
            zorder=6
        )
        
    # Physik-Regionen markieren
    ax.axvline(x=0.25, color='#ff0055', linestyle='--', alpha=0.5, lw=1.2)
    ax.text(0.05, 1.2, 'CHAOTISCHER KOLLAPS-KORRIDOR\n(Sehr geringe Zeit-Schere ΔΦ)', color='#ff5588', fontsize=9, rotation=90, alpha=0.8)

    ax.axhline(y=1.5, color='#00ff88', linestyle=':', alpha=0.4, lw=1.2)
    ax.text(1.5, 1.8, 'KAM-RESONANZFALLE\n(Starker Rotationsschutz η)', color='#66ffaa', fontsize=10, alpha=0.8)

    # Achsen-Beschriftungen & Titel
    ax.set_title('DAS 1-MILLIARDE-SYSTEME PHASENRAUM-DIAGRAMM (TRINITÄTS-TOPOLOGIE)\nVerteilung von 519.810.403 echten 3-Körper-Zerfällen auf der NVIDIA RTX 4070', 
                 fontsize=14, fontweight='bold', color='#f8fafc', pad=18)
    ax.set_xlabel('Gravitative Zeit-Schere ΔΦ_rel = (Φ_max − Φ_min) / Φ_mittel', fontsize=12, color='#e2e8f0', labelpad=10)
    ax.set_ylabel('Rotations-Schutzfaktor η_rot = L² / (|E| · I)', fontsize=12, color='#e2e8f0', labelpad=10)
    
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.0, 2.5)
    ax.grid(color='#1e293b', linestyle='--', alpha=0.5)
    
    out_path = "assets/phase_space_density_1b.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print(f"Diagramm erfolgreich gerendert: {out_path}")

if __name__ == "__main__":
    plot_heatmap()
