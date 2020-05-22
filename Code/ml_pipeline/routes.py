from flask import render_template, url_for, flash, redirect, request, send_from_directory
from werkzeug.utils import secure_filename

from ml_pipeline import app
from flask_autoindex import AutoIndex
from AppConfig import app_config
from ListAllJobs import ListAllJobs

from ManageJob import create_job as createjob
from ValidateConfig import check_if_valid_job_config, allowed_file
from MLPipeline import MLPipeline

from ml_pipeline.settings import APP_STATIC
import os


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
@app.route("/create-job", methods=['GET', 'POST'])
def create_job():
    if request.method == 'POST':
        user_form = request.form
        user_form_json = user_form.to_dict()

        error, config_user_dict = check_if_valid_job_config(user_form_json)

        if not error:
            if user_form.get("job_type") == "default_job":
                print("Default Job")
                job_desc = config_user_dict['job_description']
                job_id = createjob(job_desc, config_user_dict, app.root_path, None, is_default=False)
            else:
                print("User uploaded job")
                user_file = request.files['user_file']
                user_filename = None
                # # TODO handle other extensions error here
                if user_file and allowed_file(user_file.filename):
                    user_filename = secure_filename(user_file.filename)

                if user_filename != None:
                    job_desc = config_user_dict['job_description']
                    job_id = createjob(job_desc, config_user_dict, app.root_path, user_file, is_default=True)

                else:
                    error = True
                    config_user_dict = []
                    config_user_dict.append("Please upload a valid csv file")

        if error:
            flash_err_op = ""

            flash_err_op += "<ul>"
            for e in config_user_dict:
                flash_err_op += "<li>" + str(e) + "</li>"
            flash_err_op += "</ul>"
            flash(flash_err_op, "danger")

        else:
            succ_msg = "Congratulations! A job has been successfully created with job id " + job_id + "."
            succ_msg += "<br/>"
            succ_msg += "The job is not started yet, you can start the job and view status of all jobs on "
            view_jobs_link = "<a class='alert-link' href='/view-all-jobs'> view all jobs </a>"
            succ_msg += view_jobs_link

            flash(succ_msg, "success")

    return render_template('create_job.html', title="Create New Job")


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


@app.route("/view-all-dbs")
def view_all_dbs():
    return render_template('view_all_databases.html', title="View All Databases")


@app.route('/download_db/<path:filename>', methods=['GET', 'POST'])
def download_db(filename):
    dbs = os.path.join(APP_STATIC, "compound_dbs")
    fpath = os.path.join(dbs, filename)

    if os.path.exists(fpath):
        return send_from_directory(directory=dbs, filename=filename)
    else:
        flash("404: File you requested for download does not  exists", "danger")
        return redirect(url_for("create_job"))
