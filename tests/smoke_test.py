import sys
import os
import subprocess
import time
import urllib.request
import signal

# Ensure repository root is on sys.path so `import main` finds main.py reliably in CI
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 1) Quick import smoke (fail fast if syntax/import errors)
try:
    import main  # ensure module imports
    import training.collect_selfplay as cs
    try:
        import training.train_supervised as ts
    except Exception:
        ts = None
except Exception as e:
    print('Import failed:', e)
    sys.exit(2)

# 2) Start uvicorn as a subprocess (so the same code is exercised)
uvicorn_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
proc = None
try:
    proc = subprocess.Popen(uvicorn_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
except Exception as e:
    print("Failed to start uvicorn:", e)
    sys.exit(2)

# 3) Wait until /v1/health returns 200 or timeout
health_url = "http://127.0.0.1:8000/v1/health"
deadline = time.time() + 20.0  # seconds
last_exc = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(health_url, timeout=3) as r:
            body = r.read().decode('utf-8')
            print("Health check response:", body)
            print("Smoke test OK")
            # success -> terminate server and exit 0
            try:
                proc.terminate()
            except Exception:
                pass
            sys.exit(0)
    except Exception as e:
        last_exc = e
        time.sleep(0.5)

# timed out
print("Health check timed out; last error:", last_exc)
# try to get a bit of server output for debugging
try:
    if proc:
        out, _ = proc.communicate(timeout=1)
        print("uvicorn output:\n", out)
except Exception:
    pass
finally:
    if proc:
        try:
            proc.kill()
        except Exception:
            pass

sys.exit(2)
