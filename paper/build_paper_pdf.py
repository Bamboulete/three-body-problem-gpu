"""
Automatischer PDF-Generator für das wissenschaftliche Paper:
paper/trinity_topology.html -> paper/trinity_topology.pdf
Nutzt Selenium mit Edge im Headless-Modus für pixelgenaues Drucken.
"""

import os
import sys
import base64
import time

def build_pdf():
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options

    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_file = os.path.join(script_dir, "trinity_topology.html")
    pdf_file = os.path.join(script_dir, "trinity_topology.pdf")

    print(f"Lade HTML-Paper: {html_file}")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')

    driver = webdriver.Edge(options=options)
    driver.get(f"file:///{html_file.replace(os.sep, '/')}")
    
    # Warte kurz, damit MathJax alle Formeln rendern kann
    time.sleep(2.0)

    print("Rendere druckfertiges PDF via Edge Print-to-PDF...")
    pdf_data = driver.execute_cdp_cmd('Page.printToPDF', {
        'printBackground': True,
        'paperWidth': 8.27, # A4 Breite in Zoll
        'paperHeight': 11.69, # A4 Höhe in Zoll
        'marginTop': 0.8,
        'marginBottom': 0.8,
        'marginLeft': 0.8,
        'marginRight': 0.8
    })
    
    with open(pdf_file, 'wb') as f:
        f.write(base64.b64decode(pdf_data['data']))
        
    driver.quit()
    size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
    print(f"Wissenschaftliches Paper-PDF erfolgreich generiert:\n{pdf_file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    build_pdf()
