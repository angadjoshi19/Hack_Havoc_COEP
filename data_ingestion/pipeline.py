import os
import subprocess
import sys

def run_pipeline():
    """
    Orchestrates the entire data ingestion pipeline from fetching to final filtering.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        ("fetch_copernicus.py", "Data Fetching (Copernicus API)"),
        ("preprocess.py", "Data Preprocessing (xarray)"),
        ("statistical_filtering.py", "Statistical Filtering (scipy)")
    ]
    
    for script, description in scripts:
        print(f"\n{'='*50}")
        print(f"Starting Phase: {description}")
        print(f"{'='*50}")
        
        script_path = os.path.join(script_dir, script)
        try:
            subprocess.run([sys.executable, script_path], check=True)
            print(f"\n-> Phase '{description}' completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n-> Error: Phase '{description}' failed with exit code {e.returncode}.")
            print("Pipeline halted.")
            break

if __name__ == "__main__":
    print("Initializing OceanEmbed Data Ingestion Pipeline...")
    run_pipeline()
