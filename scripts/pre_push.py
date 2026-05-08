#!/usr/bin/env python3
"""
Pre-push runner: builds the extension (editable) and runs tests.
Exits non-zero to abort push on failure.

Behavior:
- Prefer Python from `vitvenv` if present (repo-root/vitvenv).
- Install editable extension (`pip install -e ./ext`) so the module is importable.
- Install `pytest` if missing.
- Run `pytest -q` (or a subset if desired).
"""
import os
import sys
import subprocess


def find_venv_python(repo_root):
    p1 = os.path.join(repo_root, 'vitvenv', 'bin', 'python')
    p2 = os.path.join(repo_root, 'vitvenv', 'Scripts', 'python.exe')
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None


def run(cmd, env=None, check=True):
    print('RUN:', ' '.join(cmd))
    res = subprocess.run(cmd, env=env)
    if check and res.returncode != 0:
        raise SystemExit(res.returncode)
    return res.returncode


def main():
    repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip()

    venv_py = find_venv_python(repo_root)
    if venv_py:
        py = venv_py
    else:
        py = sys.executable

    # Ensure pip, setuptools, wheel are current
    run([py, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])

    # Install editable extension
    run([py, '-m', 'pip', 'install', '-e', os.path.join(repo_root, 'ext')])

    # Ensure pytest
    try:
        import pytest  # noqa: F401
    except Exception:
        run([py, '-m', 'pip', 'install', 'pytest'])

    # Run tests (you can customize to run only kernel tests)
    test_cmd = [py, '-m', 'pytest', '-q']
    return_code = subprocess.call(test_cmd)
    if return_code != 0:
        print('Tests failed with exit code', return_code)
    else:
        print('Tests passed')
    raise SystemExit(return_code)


if __name__ == '__main__':
    main()
