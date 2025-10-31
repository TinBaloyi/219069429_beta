"""
run_training.py - Cross-platform training runner
Run from project root: python run_training.py
"""
import subprocess
import sys
from pathlib import Path
import json

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def check_file(path, fallback=None):
    """Check if file exists, try fallback if provided"""
    p = Path(path)
    if p.exists():
        print(f"✓ Found: {path}")
        return str(p)
    
    if fallback:
        fp = Path(fallback)
        if fp.exists():
            print(f"⚠ Using fallback: {fallback}")
            return str(fp)
    
    print(f"✗ Not found: {path}")
    return None

def run_training():
    print_header("AI Bottleneck Detection - Model Training")
    
    # Check we're in the right place
    if not Path("pipelines/main.py").exists():
        print("ERROR: Must run from project root directory")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"✓ Running from: {Path.cwd()}")
    
    # Create directories
    Path("out").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    print("✓ Directories ready\n")
    
    # Check for data files
    logistics_data = check_file(
        "out/cleaned_logistics.csv",
        fallback="data/logistics.csv"
    )
    manufacturing_data = check_file(
        "out/cleaned_manufacturing.csv",
        fallback="data/manufacturing.csv"
    )
    
    if not logistics_data or not manufacturing_data:
        print("\nERROR: Required data files not found!")
        sys.exit(1)
    
    print("✓ All data files located\n")
    
    # Train logistics
    print_header("Training Logistics Models")
    try:
        result = subprocess.run([
            sys.executable,
            "models/train_logistics_eta.py",
            "--train_csv", logistics_data,
            "--models_dir", "models",
            "--pred_csv", logistics_data,
            "--pred_out", "out/logistics_preds.csv"
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print("\n✓ Logistics models trained successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Logistics training failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)
    
    # Train manufacturing
    print_header("Training Manufacturing Models")
    try:
        result = subprocess.run([
            sys.executable,
            "models/train_manufacturing_speed_and_error.py",
            "--train_csv", manufacturing_data,
            "--models_dir", "models",
            "--pred_csv", manufacturing_data,
            "--pred_out", "out/mf_preds.csv"
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print("\n✓ Manufacturing models trained successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Manufacturing training failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)
    
    # Summary
    print_header("Training Complete!")
    
    print("Generated Models:")
    models = list(Path("models").glob("*.joblib"))
    if models:
        for m in models:
            print(f"  - {m.name} ({m.stat().st_size / 1024:.1f} KB)")
    else:
        print("  No .joblib models found")
    
    print("\nModel Cards:")
    cards = list(Path("models").glob("*_modelcard.json"))
    if cards:
        for c in cards:
            print(f"  - {c.name}")
            # Try to load and show key metrics
            try:
                with open(c) as f:
                    data = json.load(f)
                    if "metrics" in data and "test" in data["metrics"]:
                        test_metrics = data["metrics"]["test"]
                        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in test_metrics.items()])
                        print(f"    Test metrics: {metrics_str}")
            except:
                pass
    else:
        print("  No model cards found")
    
    print("\nPredictions:")
    preds = list(Path("out").glob("*_preds.csv"))
    if preds:
        for p in preds:
            rows = sum(1 for _ in open(p)) - 1  # Minus header
            print(f"  - {p.name} ({rows:,} predictions)")
    else:
        print("  No prediction files found")
    
    print("\nReports:")
    reports = list(Path("out").glob("*_report.html"))
    if reports:
        for r in reports:
            print(f"  - {r.name}")
    else:
        print("  No HTML reports found")
    
    print("\n" + "="*60)
    print("✓ All training completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_training()