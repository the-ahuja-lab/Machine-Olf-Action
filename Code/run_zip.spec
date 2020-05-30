# -*- mode: python ; coding: utf-8 -*-
import sys
from os import path

site_packages = next(p for p in sys.path if 'site-packages' in p)
print("site_packages",site_packages)
block_cipher = None

added_files =[
    ('ml_pipeline/templates', 'templates'),
    ('ml_pipeline/static', 'static'),
	('ml_pipeline/static', 'ml_pipeline/static'),
	(os.path.join(site_packages,'mordred'), 'mordred')
    ]

print("Add folders/files", 	added_files)

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

#flask_autoindex_hiddenimports = collect_submodules('flask_autoindex')
#flask_silk_hiddenimports = collect_submodules('flask_silk')
padel_py_hiddenimports = collect_submodules('padelpy')
#mordred_py_hiddenimports = collect_submodules('mordred')

# imports for making mordred work
networkx_py_hiddenimports = collect_submodules('networkx')
rdkitchem_py_hiddenimports = collect_submodules('rdkit.Chem')


flask_autoindex_datafiles = collect_data_files('flask_autoindex')
flask_silk_datafiles = collect_data_files('flask_silk')
padel_py_datafiles = collect_data_files('padelpy')
#mordred_py_datafiles = collect_data_files('mordred')


all_hiddenimports = ['pkg_resources.py2_warn','sklearn.utils._cython_blas','sklearn.neighbors.typedefs','sklearn.neighbors.quad_tree','sklearn.tree._utils']
#all_hiddenimports.extend(flask_autoindex_hiddenimports)
#all_hiddenimports.extend(flask_silk_hiddenimports)
all_hiddenimports.extend(padel_py_hiddenimports)
#all_hiddenimports.extend(mordred_py_hiddenimports)
all_hiddenimports.extend(networkx_py_hiddenimports)
all_hiddenimports.extend(rdkitchem_py_hiddenimports)


added_files.extend(flask_autoindex_datafiles)
added_files.extend(flask_silk_datafiles)
added_files.extend(padel_py_datafiles)
#added_files.extend(mordred_py_datafiles)


#print("@@ all_data_files ", added_files)
#print("@@ all_hiddenimports ", all_hiddenimports)

a = Analysis(['run.py'],
             pathex=['.', 'ml_pipeline/model'],
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
          name='run',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='run')
