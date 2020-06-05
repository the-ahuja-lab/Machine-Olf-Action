from flask import render_template, url_for, flash, redirect, request, send_from_directory
from werkzeug.utils import secure_filename

from pebble import concurrent
from concurrent.futures._base import CancelledError

from ml_pipeline import app, running_jobs_details
import ml_pipeline.utils.Helper as helper

from flask_autoindex import AutoIndex
from AppConfig import app_config
from ListAllJobs import ListAllJobs

from ManageJob import create_job as createjob
from ValidateConfig import check_if_valid_job_config, allowed_file
from UpdateAppConfig import check_if_valid_app_config, update_app_config, get_app_config
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
                job_id = createjob(config_user_dict, None, is_example_job=True)
            else:
                user_file = request.files['user_file']
                user_filename = None

                if user_file and allowed_file(user_file.filename):
                    user_filename = secure_filename(user_file.filename)

                if user_filename != None:
                    job_id = createjob(config_user_dict, user_file, is_example_job=False)

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

    error = None

    print("inside start_or_resume_job, before running_jobs_details ", running_jobs_details)

    if job_id in running_jobs_details.keys():
        if running_jobs_details[job_id][0] == "Running":
            error = "Job already in running queue, cannot start it again."
        else:
            print("Status is not running, start job again")
            error = None
    elif len(running_jobs_details) >= 2:
        running_jobs = ""
        for job_id, job_details in running_jobs_details.items():
            running_jobs += job_id + ", "
            error = "Cannot run more than two jobs in parallel. Jobs with job ids {} are already running".format(
                running_jobs)
    else:
        running_jobs_details[job_id] = ("Added", None)

    print("inside start_or_resume_job, after running_jobs_details ", running_jobs_details)

    if not error is None:
        flash(error, "danger")
        redirect(url_for("view_all_jobs"))
    else:
        for job_id, job_details in running_jobs_details.items():
            if job_details[0] == 'Added':
                future = start_ml_job(job_id)
                future.add_done_callback(job_done_callback)
                running_jobs_details[job_id] = ("Running", future)
                flash(
                    "Request to start job {} sent successfully. It can take many hours to complete the job based on job configuration you selected while creating job. Visit \"View Details\" page of the job to get more information about job like its current status, result files and logs.".format(
                        job_id),
                    "success")

    return redirect(url_for("view_all_jobs"))


def job_done_callback(future):
    print("Inside job_done_callback")
    print("job_done_callback Running status ", future.running())

    try:
        job_id, job_success_status = future.result()
        print("Future result inside job_done_callback ", job_id, job_success_status)

        if job_id in running_jobs_details:
            del running_jobs_details[job_id]
    except CancelledError as ce:
        print("CancelledError Exception inside job_done_callback ", ce)
    except Exception as e:
        print("Exception inside job_done_callback ", e)

@concurrent.process(daemon=False)
def start_ml_job(job_id):
    print('[pid:%s] performing calculation on %s' % (os.getpid(), job_id))
    ml = MLPipeline(job_id)
    job_success_status = ml.start()
    return job_id, job_success_status


@app.route("/stop_running_job", methods=['GET'])
def stop_running_job():
    job_id = request.args.get('job_id')

    error = None

    if job_id in running_jobs_details.keys():
        job_status, job_future = running_jobs_details[job_id]

        print("@@@@@ ", job_status, job_future)
        if not job_future is None:
            job_run_status = job_future.running()
        print("Inside stop_running_job with job id {} and job_run_status {}".format(job_id, job_run_status))

        if job_run_status:
            job_future.cancel()
            flash(
                "Request to stop job {} sent successfully. You can resume this job at some later point in time.".format(
                    job_id),
                "success")

            # regardless of errored or sucess or failure or whatever reason, remove it from running_jobs_details map
            if job_id in running_jobs_details:
                del running_jobs_details[job_id]

                print("Updating status to file")
                helper.update_running_job_status(job_id, "Stopped")
                # TODO write stopped status here in job_status file
        else:
            error = "Job is not running, cannot stop it."
    else:
        error = "Job is not running, cannot stop it."

    if not error is None:
        flash(error, "danger")

    return redirect(url_for("view_all_jobs"))


job_files_index = AutoIndex(app, app_config['jobs_folder'])

@app.route("/view-job-files")
@app.route('/view-job-files/<path:path>')
def view_job_files(path='.'):
    print("Path ", path)
    return job_files_index.render_autoindex(path=path)


@app.route("/view-job-details")
@app.route('/view-job-details/<path:path>')
def view_job_details(path='.'):
    print("Path ", path)
    # return job_files_index.render_autoindex(path=path)
    job = None
    return render_template('view_job_detail.html', title="View Job Detail", job=job)


@app.route("/view-all-jobs")
def view_all_jobs():
    all_jobs = ListAllJobs()
    all_jobs_lst = all_jobs.get_all_jobs()
    if len(all_jobs_lst) == 0:
        flash("No jobs present yet, create a new job first and then check back here", "danger")
    return render_template('view_all_jobs.html', title="View All Jobs", all_jobs=all_jobs_lst)


@app.route("/view-all-dbs")
def view_all_dbs():
    db_paths_config = {}
    all_app_configs = get_app_config()
    if not all_app_configs is None:
        db_paths_config['pubchem_db_fld_path'] = all_app_configs['pubchem_db_fld_path']
        db_paths_config['user_db_fld_path'] = all_app_configs['user_db_fld_path']
    else:
        db_paths_config['pubchem_db_fld_path'] = ""
        db_paths_config['user_db_fld_path'] = ""
    return render_template('view_all_databases.html', title="View All Databases", all_app_configs=all_app_configs)


@app.route('/download_db/<path:filename>', methods=['GET', 'POST'])
def download_db(filename):
    dbs = os.path.join(APP_STATIC, "compound_dbs")
    fpath = os.path.join(dbs, filename)

    if os.path.exists(fpath):
        return send_from_directory(directory=dbs, filename=filename)
    else:
        flash("404: File you requested for download does not  exists", "danger")
        return redirect(url_for("create_job"))


@app.route("/update_db_paths", methods=['POST'])
def update_db_paths():
    if request.method == 'POST':
        print("Inside update_db_paths")
        db_paths_form = request.form
        db_paths_form_json = db_paths_form.to_dict()
        print(db_paths_form_json)

        error, app_config_dict = check_if_valid_app_config(db_paths_form_json)

        if not error:
            success = update_app_config(db_paths_form_json)
            if success:
                flash("Folder paths updated successfully", "success")
            else:
                flash("An error occurred while updating folder paths, please try again later", "danger")
        else:
            flash_err_op = ""

            flash_err_op += "<ul>"
            for e in app_config_dict:
                flash_err_op += "<li>" + str(e) + "</li>"
            flash_err_op += "</ul>"
            flash(flash_err_op, "danger")

        return redirect(url_for("view_all_dbs"))
