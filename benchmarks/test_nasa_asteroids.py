"""
NASA CNEOS / JPL SENTRY & CLOSE-APPROACH SCANNER
Untersuche alle von der NASA überwachten gefährlichen Asteroiden (PHAs)
und berechne für die engsten Vorbeiflüge der Menschheitsgeschichte
die Gravitative Zeit-Schere und das Doppler-Tor!
"""

import sys
import json
import math
import urllib.request
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AU_KM = 149597870.7 # km in 1 AU
G_SI = 6.67430e-11
M_SUN = 1.98847e30
M_EARTH = 5.97219e24

def fetch_top_risk_asteroids():
    print("Verbinde mit NASA JPL Sentry Impact Risk Database...")
    url = "https://ssd-api.jpl.nasa.gov/sentry.api"
    req = urllib.request.Request(url, headers={'User-Agent': 'Trinity-Asteroid-Scanner'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    
    # Sortiere nach Palermo Technical Impact Hazard Scale (ps_max) oder Impact Probability (ip)
    items = data.get('data', [])
    print(f"Erfolgreich empfangen: {len(items)} von der NASA gelistete Risiko-Asteroiden!\n")
    
    # Sortiere nach höchster Einschlags-Wahrscheinlichkeit oder Palermo-Skala
    def get_risk_score(x):
        try:
            return float(x.get('ps_max', -99))
        except:
            return -99.0
            
    items.sort(key=get_risk_score, reverse=True)
    return items

def fetch_close_approaches(min_dist_au=0.05, limit=50):
    print(f"Lade die extremsten Nahbegegnungen der NASA (Abstand < {min_dist_au} AU)...")
    url = f"https://ssd-api.jpl.nasa.gov/cad.api?dist-max={min_dist_au}&sort=dist&limit={limit}&date-min=1950-01-01&date-max=2150-01-01"
    req = urllib.request.Request(url, headers={'User-Agent': 'Trinity-Asteroid-Scanner'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    
    fields = data.get('fields', [])
    rows = data.get('data', [])
    records = [dict(zip(fields, r)) for r in rows]
    return records

def analyze_asteroid_encounter(des, fullname, date_str, dist_km, v_rel_kms, h_mag=None):
    """
    Berechne Zeit-Schere und Doppler-Tor für eine Asteroiden-Erd-Begegnung.
    """
    # Schätze Masse aus absoluter Helligkeit H (sofern vorhanden)
    # H ≈ 20 -> ~300m, H ≈ 15 -> ~3km, H ≈ 25 -> ~30m
    if h_mag is not None and float(h_mag) > 0:
        h = float(h_mag)
        diam_km = (1329.0 / math.sqrt(0.14)) * (10.0 ** (-0.2 * h))
        m_ast = (4.0/3.0) * math.pi * ((diam_km * 500.0)**3) * 2500.0 # Dichte 2.5 g/cm^3
    else:
        diam_km = 0.3
        m_ast = 6.0e10 # Default Apophis-Klasse
        
    # Abstände in Metern
    d_earth_sun = 1.0 * 1.495978707e11
    d_ast_earth = dist_km * 1000.0
    d_ast_sun = d_earth_sun + d_ast_earth # Annäherung im Perihel/Aphel
    
    # Gravitatives Potenzial
    phi_sun = G_SI * (M_EARTH / d_earth_sun + m_ast / d_ast_sun)
    phi_earth = G_SI * (M_SUN / d_earth_sun + m_ast / d_ast_earth)
    phi_ast = G_SI * (M_SUN / d_ast_sun + M_EARTH / d_ast_earth)
    
    phis = [phi_sun, phi_earth, phi_ast]
    phi_max = max(phis)
    phi_min = min(phis)
    phi_mean = np.mean(phis)
    
    delta_phi_rel = (phi_max - phi_min) / max(phi_mean, 1e-12)
    
    # Doppler-Tor: Ausrichtung der Relativgeschwindigkeit v_rel zur Erdbahngeschwindigkeit (29.78 km/s)
    # Wenn v_rel_kms parallel zur Erdbahn gerichtet ist: prograd.
    # Bei den meisten NEOs liegt v_rel zwischen 5 und 30 km/s.
    v_earth = 29.78 # km/s
    # Ein Asteroid mit hoher Relativgeschwindigkeit (> 30 km/s) kommt meist mit hohem Schnittwinkel oder retrograd!
    # cos(alpha) lässt sich aus dem Vektordreieck v_ast² = v_earth² + v_rel² - 2*v_earth*v_rel*cos(theta) schätzen:
    # Je höher v_rel relativ zu v_earth, desto steiler / retrograder ist der Schnittwinkel!
    cos_encounter = 1.0 - (v_rel_kms**2) / (2.0 * (v_earth**2) + 1e-6)
    cos_encounter = max(-1.0, min(1.0, cos_encounter))
    
    # Risikobewertung nach Trinity-Topologie
    is_critical_gate = (cos_encounter < -0.3)
    is_dangerously_close = (dist_km < 100000.0) # Unter Mondabstand (384.400 km)
    
    if dist_km < 35000.0 and cos_encounter < 0.0:
        risk_level = "🚨 EXTREM KRITISCH (Kollisions- / Keyhole-Zone!)"
    elif dist_km < 100000.0 and is_critical_gate:
        risk_level = "⚠️ HOHE GEFAHR (Starkes Doppler-Stoß-Risiko)"
    elif dist_km < 384400.0:
        risk_level = "🟡 BEOBACHTUNG (Vorbeiflug innerhalb der Mondbahn)"
    else:
        risk_level = "🟢 HARMLOS (Sicherer Abstand)"
        
    return {
        "des": des,
        "name": fullname or des,
        "date": date_str,
        "dist_km": dist_km,
        "dist_lunar": dist_km / 384400.0,
        "v_rel_kms": v_rel_kms,
        "diam_m": diam_km * 1000.0,
        "delta_phi_rel": delta_phi_rel,
        "cos_alpha": cos_encounter,
        "risk_level": risk_level
    }

def run_scanner():
    print("=" * 115)
    print("   NASA CNEOS / JPL ASTEROIDEN-SCANNER: TOPOLOGISCHE DOPPLER-TOR & ZEIT-SCHEREN-ANALYSE")
    print("=" * 115)
    
    # 1. Top NASA Sentry Risikokandidaten
    sentry_items = fetch_top_risk_asteroids()
    
    print("TOP 10 OFFIZIELLE NASA-RISIKO-ASTEROIDEN (PALERMO / SENTRY-SKALA):")
    print(f"{'Asteroid':<20} │ {'Durchmesser':<14} │ {'Einschlags-Jahre':<20} │ {'Wahrscheinlichkeit':<20} │ {'Palermo-Score':<14}")
    print("─" * 100)
    for it in sentry_items[:10]:
        name = it.get('fullname', it.get('des', 'Unbekannt'))
        diam = f"{float(it.get('diameter', 0.1))*1000:.0f} m" if it.get('diameter') else "ca. 50-200 m"
        years = it.get('range', 'N/A')
        prob = f"{float(it.get('ip', 0))*100:.4f}%"
        ps = it.get('ps_max', 'N/A')
        print(f"{name:<20} │ {diam:<14} │ {years:<20} │ {prob:<20} │ {ps:<14}")
    print("=" * 100)
    
    # 2. Reale Nahbegegnungen (Close Approaches)
    print("\nUNTERSUCHE DIE ENGSTEN HISTORISCHEN & ZUKÜNFTIGEN VORBEIFLÜGE AN DER ERDE:")
    approaches = fetch_close_approaches(min_dist_au=0.005, limit=20) # Extrem nah: < 0.005 AU (~750.000 km)
    
    results = []
    for app in approaches:
        des = app.get('des')
        date_str = app.get('cd')
        dist_km = float(app.get('dist')) * AU_KM
        v_rel = float(app.get('v_rel'))
        h = app.get('h')
        res = analyze_asteroid_encounter(des, des, date_str, dist_km, v_rel, h)
        results.append(res)
        
    print(f"{'Asteroid / Objekt':<20} │ {'Datum Vorbeiflug':<18} │ {'Abstand (km)':<15} │ {'Monddistanz':<13} │ {'Doppler cos α':<15} │ {'Risiko-Status':<35}")
    print("─" * 125)
    for r in results[:15]:
        dist_str = f"{r['dist_km']:,.0f} km"
        ld_str = f"{r['dist_lunar']:.2f} LD"
        dop_str = f"{r['cos_alpha']:+.3f}"
        print(f"{r['des']:<20} │ {r['date']:<18} │ {dist_str:<15} │ {ld_str:<13} │ {dop_str:<15} │ {r['risk_level']:<35}")
        
    print("=" * 125)
    
    # 3. Berühmte Spezial-Kandidaten (Bennu, 1950 DA, Apophis)
    print("\nDIE 3 BERÜHMTESTEN RISIKO-KANDIDATEN IM DOPPLER-TOR-VERGLEICH:")
    special = [
        {"name": "99942 Apophis", "date": "2029-04-13", "dist_km": 31600.0, "v_rel": 7.43, "h": 19.7, "desc": "Rast unter Satelliten durch (31.600 km). Tor 2029 zu, Keyhole für 2036 gebannt."},
        {"name": "101955 Bennu", "date": "2182-09-24", "dist_km": 120000.0, "v_rel": 6.30, "h": 20.6, "desc": "Ziel von NASA OSIRIS-REx. Höchstes kumulatives Risiko (1:2700 im Jahr 2182)."},
        {"name": "(29075) 1950 DA", "date": "2880-03-16", "dist_km": 80000.0, "v_rel": 14.10, "h": 17.0, "desc": "Riesiger 1.3-km-Asteroid. Potenzieller Zivilisations-Killer im Jahr 2880."}
    ]
    
    for s in special:
        res = analyze_asteroid_encounter(s["name"], s["name"], s["date"], s["dist_km"], s["v_rel"], s["h"])
        print(f"\n• {s['name']} (Begegnung am {s['date']}):")
        print(f"  - Abstand: {s['dist_km']:,} km ({res['dist_lunar']:.2f} Monddistanzen) | Relativtempo: {s['v_rel']} km/s | Durchmesser: {res['diam_m']:.0f} m")
        print(f"  - Doppler-Tor cos(α): {res['cos_alpha']:+.4f} | Zeit-Schere ΔΦ: {res['delta_phi_rel']:.4f}")
        print(f"  - Analyse: {s['desc']}")
        print(f"  - Einstufung: {res['risk_level']}")

if __name__ == "__main__":
    run_scanner()
