from flask import Flask
import os
import sys

base_dir = '.'
if hasattr(sys, '_MEIPASS'):
    base_dir = os.path.join(sys._MEIPASS)
    print("Inside _MEIPASS base_dir before ", base_dir)
    base_dir = sys._MEIPASS
    print("After _MEIPASS base_dir before ", base_dir)

app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

from ml_pipeline import routes
