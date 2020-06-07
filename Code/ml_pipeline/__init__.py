from flask import Flask
import logging

import os
import sys

# disable info logs of flask
logger_flask = logging.getLogger('werkzeug')
logger_flask.setLevel(logging.ERROR)

# setting base directory of application as current directory
base_dir = '.'

# changing basedir when using pyinstaller's onefile packaging
if hasattr(sys, '_MEIPASS'):
    base_dir = os.path.join(sys._MEIPASS)
    base_dir = sys._MEIPASS

app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))

app.config['SECRET_KEY'] = '69a4924601418746dc6c4d536339b5b3'
# to allow change in css and js to reflect immediately
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

running_jobs_details = {}
log_viewer_app = {}

from ml_pipeline import routes
