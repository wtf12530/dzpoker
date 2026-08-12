import sys
# Smoke test: import key modules and run simple functions
try:
    import main
    import training.collect_selfplay as cs
    import training.train_supervised as ts
    print('Import OK')
    sys.exit(0)
except Exception as e:
    print('Import failed:', e)
    sys.exit(2)
