# Task7 Cache Integer IDs

`pvad_cache` now accepts native JSON integer Dataset-A IDs and canonicalizes them to decimal strings at input validation. All cache ordering, digest, coverage, feature, and resume-path operations therefore use the canonical string form.

The boundary remains fail-closed: booleans, floats, empty IDs, traversal IDs, and integer/string duplicates are rejected.

Verification completed without running full Dataset-A:

- `pytest tests/test_pvad_cache.py tests/test_firered_model_assets.py`: 135 passed, 3 skipped
- `python -m py_compile xh202615/pvad_cache.py tests/test_pvad_cache.py tests/test_firered_model_assets.py`
