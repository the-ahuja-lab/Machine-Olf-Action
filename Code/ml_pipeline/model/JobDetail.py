from AppConfig import app_config
import os
from datetime import datetime
import json

class JobDetail:
    def __init__(self):
        self.all_jobs_fld = app_config['jobs_folder']
