Git hooks
---------

To run tests automatically before pushing, install the repo hooks:

Unix / Git Bash:

```bash
./scripts/install_hooks.sh
```

Windows PowerShell:

```powershell
./scripts/install_hooks.ps1
```

The pre-push hook will build the extension (editable install) and run `pytest -q`. If tests fail, the push is aborted.
