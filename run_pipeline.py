"""
================================================================================
 ██████╗ ██╗     ██████╗ ██╗  ██╗ █████╗     ██████╗ ███████╗███████╗███████╗ ██████╗████████╗
██╔═══██╗██║     ██╔══██╗██║  ██║██╔══██╗    ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝╚══██╔══╝
███████║██║     ██████╔╝███████║███████║    ██║  ██║█████╗  █████╗  █████╗  ██║        ██║
██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║    ██║  ██║██╔══╝  ██╔══╝  ██╔══╝  ██║        ██║
██║  ██║███████╗██║     ██║  ██║██║  ██║    ██████╔╝███████╗██║     ███████╗╚██████╗   ██║
╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝

  MASTER PIPELINE — Alpha Defect Detection in Hot Rolling Mills
  ═══════════════════════════════════════════════════════════════
  
  World-class, production-grade ML pipeline for rare defect detection.
  
  Business Constraint: FALSE NEGATIVES ARE UNACCEPTABLE
  Primary Objective : RECALL = 100%, Precision > 90%
  
  Pipeline Phases:
    1. Exploratory Data Analysis
    2. Advanced Feature Engineering
    3-4. Model Training (8 diverse models)
    5. Optuna Hyperparameter Optimization
    6-7. Threshold Engineering & Ensemble System
    8. SHAP Explainability & Error Analysis
    
  Author: AI Industrial Defect Detection Specialist
  Version: 1.0 — Production Grade
================================================================================
"""

import time
import sys
import os
import traceback

sys.path.insert(0, r"d:\Tata_steel")

PHASES = [
    ("Phase 1 — Exploratory Data Analysis",       "phase1_eda"),
    ("Phase 2 — Advanced Feature Engineering",     "phase2_feature_engineering"),
    ("Phase 3 & 4 — Model Training",              "phase3_4_model_training"),
    ("Phase 5 — Optuna Hyperparameter Tuning",    "phase5_optuna_tuning"),
    ("Phase 6 & 7 — Ensemble & Threshold",        "phase6_7_ensemble"),
    ("Phase 8 — Explainability",                  "phase8_explainability"),
]

def run_pipeline():
    """Execute the full pipeline sequentially."""
    start_total = time.time()
    results = {}
    
    print("\n" + "█" * 80)
    print("  ALPHA DEFECT DETECTION — MASTER PIPELINE")
    print("  Hot Rolling Mill Quality Assurance System")
    print("█" * 80)
    
    for phase_name, module_name in PHASES:
        print(f"\n\n{'▓' * 80}")
        print(f"  STARTING: {phase_name}")
        print(f"{'▓' * 80}")
        
        start = time.time()
        try:
            # Dynamic import and execution
            module = __import__(module_name)
            elapsed = time.time() - start
            results[module_name] = {"status": "SUCCESS", "time": elapsed}
            print(f"\n  ⏱️  {phase_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            results[module_name] = {"status": "FAILED", "time": elapsed, "error": str(e)}
            print(f"\n  ❌ {phase_name} FAILED after {elapsed:.1f}s")
            print(f"  Error: {e}")
            traceback.print_exc()
            # Continue to next phase (some phases may work independently)
            continue
    
    total_time = time.time() - start_total
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n\n" + "█" * 80)
    print("  PIPELINE EXECUTION SUMMARY")
    print("█" * 80)
    
    for phase_name, module_name in PHASES:
        r = results.get(module_name, {"status": "SKIPPED", "time": 0})
        status = "✅" if r["status"] == "SUCCESS" else "❌"
        print(f"  {status} {phase_name:45s} — {r['time']:7.1f}s — {r['status']}")
    
    print(f"\n  Total Pipeline Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("█" * 80)
    
    # ── Final check ──────────────────────────────────────────────────────────
    expected_sub = os.path.join(r"d:\Tata_steel", "expected_submission.csv")
    if os.path.exists(expected_sub):
        import pandas as pd
        sub = pd.read_csv(expected_sub)
        print(f"\n  📊 Final Submission: {len(sub)} rows, {sub['Y'].sum()} predicted defects")
        print(f"  📁 File: {expected_sub}")
    
    return results


if __name__ == "__main__":
    results = run_pipeline()
