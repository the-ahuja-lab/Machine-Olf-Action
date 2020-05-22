import sys
import os

from flask import render_template, url_for, flash, redirect, request
from werkzeug.utils import secure_filename

from ml_pipeline import app
from flask_autoindex import AutoIndex
from AppConfig import app_config
from ListAllJobs import ListAllJobs

# print("Before sys path ", sys.path)
#
# base_path = app.root_path
#
# print("Base path ", base_path)
# sys.path.append(os.path.join(base_path, "model"))
#
# print("After sys path ", sys.path)

from ManageJob import create_job as createjob
from ValidateConfig import check_if_valid_job_config, allowed_file
from MLPipeline import MLPipeline


#
# @app.route("/")
# @app.route("/home")
# def home():
#     # return render_template(url_for(create_job))
#     return render_template('landing.html')


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
@app.route("/create-job", methods=['GET', 'POST'])
def create_job():
    # parse json file and input
    # if both input valid, show success message, create new job folder, upload files to correct location
    # otherwise show error message
    if request.method == 'POST':
        user_form = request.form
        print("@@@@ result", user_form)
        # print(user_form.to_dict())
        # print(user_form.to_dict(flat=False))
        # json_data = request.get_json(force=True)
        #
        # print("@@ json_data ", json_data)
        user_form_json = user_form.to_dict()
        config_user_dict = check_if_valid_job_config(user_form_json)

        if user_form.get("job_type") == "default_job":
            print("Default Job")
            if config_user_dict != None:
                job_desc = config_user_dict['job_description']
                job_id = createjob(job_desc, config_user_dict, app.root_path, None, is_default=False)
        else:
            print("User uploaded job")
            user_file = request.files['user_file']
            user_filename = None
            # # TODO handle other extensions error here
            if user_file and allowed_file(user_file.filename):
                user_filename = secure_filename(user_file.filename)

            if config_user_dict != None and user_filename != None:
                job_desc = config_user_dict['job_description']
                job_id = createjob(job_desc, config_user_dict, app.root_path, user_file, is_default=True)

        # user_file = request.files['user_file']
        # user_filename = None
        # if user_file and allowed_file(user_file.filename):
        #     user_filename = secure_filename(user_file.filename)
        # # TODO handle other extensions error here
        # config_user_dict = check_if_valid_job_config(user_form_json)
        #
        # if config_user_dict != None and user_filename != None:
        #     job_desc = config_user_dict['job_description']
        #     job_id = createjob(job_desc, config_user_dict, app.root_path, user_file, is_default=True)

        # return redirect(url_for('view_all_jobs'))
    # else:
    return render_template('create_job.html', title="Create New Job")


# @app.route("/create_example_job", methods=['POST'])
# def create_example_job():
#     # parse json file and input
#     # if both input valid, show success message, create new job folder, upload files to correct location
#     # otherwise show error message
#     if request.method == 'POST':
#         user_form = request.form
#         print("@@@@ result", user_form)
#         # print(user_form.to_dict())
#         # print(user_form.to_dict(flat=False))
#         # json_data = request.get_json(force=True)
#         #
#         # print("@@ json_data ", json_data)
#         user_form_json = user_form.to_dict()
#
#         config_user_dict = check_if_valid_job_config(user_form_json)
#
#         if config_user_dict != None:
#             job_desc = config_user_dict['job_description']
#             job_id = createjob(job_desc, config_user_dict, app.root_path, None, is_default=True)
#
#             # return redirect(url_for('view_all_jobs'))
#     # else:
#     return render_template('create_job.html', title="Create New Job")

# @app.route("/create_job_service", methods=['GET', 'POST'])
# def create_job_service():
#     # parse json file and input
#     # if both input valid, show success message, create new job folder, upload files to correct location
#     # otherwise show error message
#     with app.open_resource("static/ml_ip_config.json") as f:
#         json_str = f.read()
#         if check_if_valid_job_config(json_str):
#             job_id = createjob("Default Job", json_str, app.root_path)
#             # start_or_resume_job(job_id)
#     return render_template('create_job_success.html', job_details=job_id)


@app.route("/start_job", methods=['GET'])
def start_or_resume_job():
    job_id = request.args.get('job_id')

    # TODO create a new process or thread here and update its details to job folder, if already created don't adhere request
    ml = MLPipeline(job_id)
    ml.start()

    # TODO add check job status

    return "Job Completed Successfully"


job_files_index = AutoIndex(app, app_config['jobs_folder'])


@app.route("/view-job-files")
@app.route('/view-job-files/<path:path>')
def view_job_files(path='.'):
    print("Path ", path)
    return job_files_index.render_autoindex(path=path)


@app.route("/view-all-jobs")
def view_all_jobs():
    all_jobs = ListAllJobs()
    all_jobs_lst = all_jobs.get_all_jobs()
    return render_template('view_all_jobs.html', title="View All Jobs", all_jobs=all_jobs_lst)
