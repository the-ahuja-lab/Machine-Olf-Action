import multiprocessing
import random, threading, webbrowser
import os

from ml_pipeline import app

if __name__ == '__main__':
    multiprocessing.freeze_support()

    port = 5000 + random.randint(0, 999)
    # port = 5000
    url = "http://127.0.0.1:{0}".format(port)

    # to open default browser automatically
    if 'WERKZEUG_RUN_MAIN' not in os.environ:
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    app.run(port=port, debug=False)

    # to allow change in css and js to reflect immediately
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
