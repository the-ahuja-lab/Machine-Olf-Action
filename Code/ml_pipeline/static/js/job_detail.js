function getUrlParameter(sParam) {
    var sPageURL = window.location.search.substring(1),
        sURLVariables = sPageURL.split('&'),
        sParameterName,
        i;

    for (i = 0; i < sURLVariables.length; i++) {
        sParameterName = sURLVariables[i].split('=');

        if (sParameterName[0] === sParam) {
            return sParameterName[1] === undefined ? true : decodeURIComponent(sParameterName[1]);
        }
    }
};

$(function() {
    fetch_job_details();

    show_job_details();
})

function fetch_job_details() {
    if (typeof job_run_status != 'undefined' && job_run_status != "Running") {
        console.log("Job not running, clearing interval")
        clearInterval(job_det_interval);
    }

    console.log("Inside fetch_job_details")
    $.ajax({
        url: "/fetch_job_details",
        type: "GET",
        data: {
            job_id: getUrlParameter('job_id')
        },
        success: function(response) {
            $("#job_details_div").html(response);
        },
        error: function(xhr) {
            //Do Something to handle error
            console.log("error from ajax call in fetch_job_details")
        }
    });

    job_run_status = $("#jrs_div").html()
    console.log("job_run_status after fetch_job_details ", job_run_status)
}

function show_job_details() {
    console.log("Inside show_job_details")
    $.ajax({
        url: "/show_job_logs",
        type: "GET",
        data: {
            job_id: getUrlParameter('job_id'),
            lt: getUrlParameter('lt')
        },
        success: function(response) {
            $("#job_logs_div").html(response);
        },
        error: function(xhr) {
            //Do Something to handle error
            console.log("error from ajax call in show_job_logs")
        }
    });
}

//refresh every 10 seconds the job related details, reset it inside fetch_job_details if job not running to save
//unwanted get requests to server
job_det_interval = setInterval(fetch_job_details, 10000);