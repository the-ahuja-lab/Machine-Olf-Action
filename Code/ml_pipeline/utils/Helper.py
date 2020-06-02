import json
from collections import namedtuple


def update_job_status(job_status_fp, status):
    with open(job_status_fp, mode='r') as f:
        othr_config = json.load(f)

        othr_config['status'] = status

    with open(job_status_fp, mode='w') as f:
        json.dump(othr_config, f, ensure_ascii=False, indent=4)


def _json_object_hook(d):
    return namedtuple('x', d.keys())(*d.values())


def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)


def create_job_config_object(str):
    config = json2obj(str)
    return config


def change_ext(fname, ext_init, ext_fin):
    if fname.endswith(ext_init):
        return fname.replace(ext_init, ext_fin)
    else:
        return fname + ext_fin

def infer_th_from_file_name(fname, sim_metric, ext):
    sm_end_index = fname.index(sim_metric) + len(sim_metric)
    ext_start_index = fname.index(ext)
    return fname[sm_end_index: ext_start_index]
