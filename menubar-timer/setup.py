from setuptools import setup

APP = ['timer.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'LSUIElement': True,
        'CFBundleName': 'Days Until',
        'CFBundleDisplayName': 'Days Until',
        'CFBundleGetInfoString': "Countdown timer in menu bar",
        'CFBundleIdentifier': "com.daysuntil.app",
        'CFBundleVersion': "1.0.0",
        'CFBundleShortVersionString': "1.0.0",
    },
    'packages': ['rumps'],
    'includes': ['rumps'],
    'site_packages': True,
}

setup(
    name='Days Until',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
) 