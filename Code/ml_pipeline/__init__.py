from flask import Flask
import os
import sys

# setting base directory of application as current directory
base_dir = '.'

# changing basedir when using pyinstaller's onefile packaging
if hasattr(sys, '_MEIPASS'):
    base_dir = os.path.join(sys._MEIPASS)
    base_dir = sys._MEIPASS

app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))
app.config['SECRET_KEY'] = '69a4924601418746dc6c4d536339b5b3'

from ml_pipeline import routes
