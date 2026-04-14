"""
run_nb00.py — Environment verification for RBFin Offline RL project.

Checks all required package imports, tests BCB API, yfinance, and OpenAlex.
Saves results to outputs/models/environment_check.json.
"""

import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def check_import(module_name: str) -> dict:
    """Check if a module can be imported and return its version."""
    result = {"module": module_name, "status": "ok", "version": None, "error": None}
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", None)
        if version is None:
            # Try common version attributes
            for attr in ("version", "VERSION", "__VERSION__"):
                v = getattr(mod, attr, None)
                if v is not None:
                    version = str(v)
                    break
        result["version"] = version
        print(f"  [OK] {module_name} == {version}")
    except ImportError as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [FAIL] {module_name}: {e}")
    return result


def test_bcb() -> dict:
    """Test BCB SGS API by fetching SELIC meta series (432)."""
    result = {"test": "bcb_sgs", "status": "ok", "rows": None, "error": None}
    try:
        from bcb import sgs
        data = sgs.get(432, start="2024-01-01", end="2024-03-31")
        n = len(data)
        result["rows"] = n
        print(f"  [OK] BCB SGS code 432 (SELIC meta): {n} rows returned")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [FAIL] BCB SGS: {e}")
    return result


def test_yfinance() -> dict:
    """Test yfinance by downloading 5 days of PETR4.SA."""
    result = {"test": "yfinance_petr4", "status": "ok", "rows": None, "error": None}
    try:
        import yfinance as yf
        df = yf.download("PETR4.SA", period="5d", progress=False, auto_adjust=True)
        n = len(df)
        result["rows"] = n
        print(f"  [OK] yfinance PETR4.SA: {n} rows returned")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [FAIL] yfinance: {e}")
    return result


def test_openalex() -> dict:
    """Test OpenAlex API (no key needed)."""
    result = {"test": "openalex", "status": "ok", "total_count": None, "error": None}
    try:
        import urllib.request
        import urllib.parse

        url = "https://api.openalex.org/works?search=offline+reinforcement+learning&per-page=1"
        req = urllib.request.Request(url, headers={"User-Agent": "rbfin-project/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
        total = body.get("meta", {}).get("count", None)
        result["total_count"] = total
        print(f"  [OK] OpenAlex: {total} works found for 'offline reinforcement learning'")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [FAIL] OpenAlex: {e}")
    return result


def main() -> None:
    print("=" * 60)
    print("NB-00: Environment Check")
    print(f"Python: {sys.version}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    # --- Package imports ---
    packages = [
        "torch",
        "gymnasium",
        "d3rlpy",
        "yfinance",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "ipywidgets",
        "dotenv",
        "bcb",
    ]

    print("\n[1] Checking imports...")
    import_results = [check_import(pkg) for pkg in packages]

    # --- API tests ---
    print("\n[2] Testing BCB API...")
    bcb_result = test_bcb()

    print("\n[3] Testing yfinance...")
    yf_result = test_yfinance()

    print("\n[4] Testing OpenAlex API...")
    oa_result = test_openalex()

    # --- Summarize ---
    all_ok_imports = all(r["status"] == "ok" for r in import_results)
    all_ok_apis = all(
        r["status"] == "ok" for r in [bcb_result, yf_result, oa_result]
    )

    summary = {
        "python_version": sys.version,
        "project_root": str(PROJECT_ROOT),
        "imports": import_results,
        "api_tests": [bcb_result, yf_result, oa_result],
        "all_imports_ok": all_ok_imports,
        "all_apis_ok": all_ok_apis,
        "overall_status": "PASS" if (all_ok_imports and all_ok_apis) else "PARTIAL",
    }

    output_path = OUTPUTS_DIR / "environment_check.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"All imports OK : {all_ok_imports}")
    print(f"All API tests OK: {all_ok_apis}")
    print(f"Overall status  : {summary['overall_status']}")
    print(f"Saved to        : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
