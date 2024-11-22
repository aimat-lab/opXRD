# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

import os
src_dir =os.path.dirname(os.path.abspath(__name__))
root_dir = os.path.dirname(src_dir)
print(f'root dir: {root_dir}')
print(f'src dir: {src_dir}')

a = Analysis(
    ['run/prod_run.py'],
    pathex=[root_dir],
    binaries=[],
    datas=[
        (os.path.join(src_dir, 'resources/images/*'), 'images'),
        (os.path.join(src_dir, 'resources/documents/*'), 'documents'),
    ],
    hiddenimports=['python-intervals'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='prod_run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
