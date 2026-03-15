# PyInstaller spec — cross-platform (Windows + Linux)
# Build: pyinstaller fbt_server.spec
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

IS_WINDOWS = sys.platform == "win32"

a = Analysis(
    ['gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Include mediapipe model data
        *collect_data_files('mediapipe'),
    ],
    hiddenimports=[
        'camera',
        'filter',
        'fusion',
        'calibration',
        'osc_output',
        'debug_http',
        'platform_backend',
        'flask',
        'pythonosc',
        'mediapipe',
        'cv2',
        'numpy',
        'scipy',
        *collect_submodules('mediapipe'),
        *collect_submodules('flask'),
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'wx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

if IS_WINDOWS:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='FBTServer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,        # No console window on Windows (GUI only)
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='icon.ico' if os.path.exists('icon.ico') else None,
    )
else:
    # Linux: single-file executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='FBTServer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,         # Keep console on Linux for log output
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
    )
