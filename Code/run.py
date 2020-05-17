from ml_pipeline import app
import random, threading, webbrowser

if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)

    threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    app.run(port=port, debug=False)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
