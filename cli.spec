# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['cli.py'],
             pathex=['D:\\betsi-windows'],
             binaries=[],
             datas=[],
	     hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['statsmodels'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\statsmodels", prefix="statsmodels")
a.datas += Tree("D:\\anaconda\Lib\\site-packages\\numpy", prefix="numpy")
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\pandas", prefix="pandas")
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\scipy", prefix="scipy")
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\seaborn", prefix="seaborn")
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\pathlib2", prefix="pathlib")
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\matplotlib", prefix="matplotlib")
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\PyQt5", prefix="pyqt")
a.datas += Tree("D:\\anaconda\\Lib\\site-packages\\patsy", prefix="patsy")


pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='cli',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
