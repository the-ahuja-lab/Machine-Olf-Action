var timer_info
var timer_debug
var timer_error

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
    console.log("Inside ready function of log viewer")

    job_id = getUrlParameter('job_id')
    lt = getUrlParameter('lt')

    create_ws(job_id, lt)
})

function create_ws(job_id, lt) {

    host = window.location.hostname
    protocol = "ws"
    port = "8765"
    target = "/logview"

    ws_host = protocol + "://" + host + ":" + port + target

    ws_url = ws_host + "?job_id=" + job_id + "&lt=" + lt

    console.log("ws_url ", ws_url)
    console.log("lt ", lt)

    job_log_type = "Job Info Log"
    if (lt == "error")
        job_log_type = "Job Error Log"
    else if (lt == "debug")
        job_log_type = "Job Debug Log"

    console.log("job_log_type ", job_log_type)

    $('#job_log_title').html(job_log_type)

    ws = new WebSocket(ws_url)
    $('#log_data').html("");

    ws.onerror = function(event) {
        console.log("Inside onerror event of websocket")
        $('#log_data').html("Please wait while logs are being fetched. In case they don't show up at all, try refreshing page to see if it fixes the problem.");
    };

    ws.onclose = function() {
        // Try to reconnect in 5 seconds
        console.log("Inside on close event of websocket")
        ws = null
        if(lt == "info" && get_sel_log_level() == "info"){
            clear_timeout(timer_error)
            clear_timeout(timer_debug)

            timer_info = setTimeout(function() {
                create_ws(job_id, lt)
            }, 5000);
        }else if(lt == "error" && get_sel_log_level() == "error"){
            clear_timeout(timer_info)
            clear_timeout(timer_debug)

            timer_error = setTimeout(function() {
                create_ws(job_id, lt)
            }, 5000);

        }else if(lt == "debug" && get_sel_log_level() == "debug"){
            clear_timeout(timer_info)
            clear_timeout(timer_error)

            timer_debug = setTimeout(function() {
                create_ws(job_id, lt)
            }, 5000);
        }
    };

    ws.onmessage = (event) => {
        //        log_data_html = event.data.replace(/(?:\r\n|\r|\n)/g, '<br>');
        if (event.data == "ping") {
            ws.send('pong');
            console.log('Keeping connection active, sending pong for ping from server')
        } else {
            $('#log_data').append(event.data);
            $('#card-log-body').scrollTop($('#card-log-body')[0].scrollHeight);
        }
    };
}

function change_log_level(lt) {
    console.log(lt.value);
    job_id = getUrlParameter('job_id')

    create_ws(job_id, lt.value)

}

function clear_timeout(timer){
    if (typeof timer != 'undefined')
        clearTimeout(timer);
}

function get_sel_log_level(){
    lt = $("#log_type_sel").val()
    console.log("get_sel_log_level ", lt)
    return lt
}