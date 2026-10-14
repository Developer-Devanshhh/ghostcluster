"""Run the complete analysis: python scripts/run_all.py"""
import os
import subprocess
import sys

STEPS = ["01_embed.py", "02_evaluate.py", "03_clusters.py", "04_robustness.py", "05_sensitivity.py",
         "06_hidden_concepts.py", "07_figures_tables.py"]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        subprocess.run([sys.executable, os.path.join(here, step)], check=True)
