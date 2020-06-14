# -*- mode: python ; coding: utf-8 -*-
import sys
from os import path
import platform
from pathlib import Path

def get_os_type():
    os = platform.system()
    os_lower = os.lower()
    print("OS is {}".format(os))
    return os_lower

os_type = get_os_type()
ico_path = "ml_pipeline/static/images/ml_olfa_logo.ico"

site_packages = next(p for p in sys.path if 'site-packages' in p)
print("site_packages",site_packages)
block_cipher = None

added_files =[
    ('ml_pipeline/templates', 'templates'),
    ('ml_pipeline/static', 'static'),
	('ml_pipeline/static', 'ml_pipeline/static'),
	(os.path.join(site_packages,'mordred'), 'mordred'),
    ('padelpy', 'padelpy')
	]

if os_type.startswith("windows"):
    added_files.append(("jre8/win","jre8/win"))
    ico_path = "ml_pipeline/static/images/ml_olfa_logo.ico"
elif os_type.startswith("darwin"):
    added_files.append(("jre8/mac","jre8/mac"))
    ico_path = "ml_pipeline/static/images/ml_olfa_logo.icns"
elif os_type.startswith("linux"):
    lib_abs_path = Path(site_packages).parent.parent
    print("lib_abs_path linux ", lib_abs_path)
    added_files.append((os.path.join(lib_abs_path,'libiomp5.so'), '.'))
    added_files.append((os.path.join(lib_abs_path,'libpython3.6m.so'),'.'))
    added_files.append(("jre8/linux","jre8/linux"))
    ico_path = "ml_pipeline/static/images/ml_olfa_logo.png"


print("Add folders/files", 	added_files)

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

#flask_autoindex_hiddenimports = collect_submodules('flask_autoindex')
#flask_silk_hiddenimports = collect_submodules('flask_silk')
#padel_py_hiddenimports = collect_submodules('padelpy')
#mordred_py_hiddenimports = collect_submodules('mordred')

# imports for making mordred work
networkx_py_hiddenimports = collect_submodules('networkx')
rdkitchem_py_hiddenimports = collect_submodules('rdkit.Chem')

flask_autoindex_datafiles = collect_data_files('flask_autoindex')
flask_silk_datafiles = collect_data_files('flask_silk')
#padel_py_datafiles = collect_data_files('padelpy')
#mordred_py_datafiles = collect_data_files('mordred')


all_hiddenimports = ['pkg_resources.py2_warn','sklearn.utils._cython_blas','sklearn.neighbors.typedefs','sklearn.neighbors.quad_tree','sklearn.tree._utils']
#all_hiddenimports.extend(flask_autoindex_hiddenimports)
#all_hiddenimports.extend(flask_silk_hiddenimports)
#all_hiddenimports.extend(padel_py_hiddenimports)
#all_hiddenimports.extend(mordred_py_hiddenimports)
all_hiddenimports.extend(networkx_py_hiddenimports)
all_hiddenimports.extend(rdkitchem_py_hiddenimports)

added_files.extend(flask_autoindex_datafiles)
added_files.extend(flask_silk_datafiles)
#added_files.extend(padel_py_datafiles)
#added_files.extend(mordred_py_datafiles)


#print("all_data_files ", added_files)
#print("all_hiddenimports ", all_hiddenimports)

a = Analysis(['run.py'],
             pathex=['.', 'ml_pipeline/model','padelpy'],
             binaries=[],
             datas=added_files,
             hiddenimports=all_hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='molfa',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True, icon=ico_path)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='molfa')
