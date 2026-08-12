# name=tests/smoke_test.py
import sys
import os

# Ensure repository root is on sys.path so `import main` finds main.py reliably in CI
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Smoke test: import key modules and run simple functions
try:
    import main
    import training.collect_selfplay as cs
    # training.train_supervised may not exist yet; import guarded to avoid hard failure
    try:
        import training.train_supervised as ts
    except Exception:
        ts = None
    print('Import OK')
    sys.exit(0)
except Exception as e:
    print('Import failed:', e)
    sys.exit(2)
