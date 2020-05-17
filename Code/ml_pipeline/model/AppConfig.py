import os
from pathlib import Path


def create_all_jobs_dir():
    print("Inside create_all_jobs_dir")
    prnt_fld = Path.home()
    fld_path = os.path.join(prnt_fld, "all_jobs")
    os.makedirs(fld_path, exist_ok=True)

    return fld_path


app_config = {
    "jobs_folder": create_all_jobs_dir()
}