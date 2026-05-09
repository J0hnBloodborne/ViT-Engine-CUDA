import os
import ctypes
import traceback

root = os.getcwd()
pyd = os.path.join(root, 'ext', 'vit_cuda.cp311-win_amd64.pyd')
print('pyd path:', pyd)
print('exists:', os.path.exists(pyd))
print('cwd:', root)
print('PATH contains CUDA?', any('CUDA' in p for p in os.environ.get('PATH', '').split(os.pathsep)))

try:
    ctypes.CDLL(pyd)
    print('ctypes load succeeded')
except Exception:
    traceback.print_exc()
