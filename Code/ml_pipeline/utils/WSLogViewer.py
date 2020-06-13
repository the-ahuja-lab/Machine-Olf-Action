# -*- coding: utf-8 -*-

import time
import os.path
import asyncio
import logging
import argparse
import websockets
from collections import deque
from urllib.parse import urlparse, parse_qs

# from ansi2html import Ansi2HTMLConverter

NUM_LINES = 1000
HEARTBEAT_INTERVAL = 15  # seconds

# init
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
# logger = logging.getLogger('websockets')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())

allowed_prefixes = []

log_folder = ""


# conv = Ansi2HTMLConverter(inline=True)

@asyncio.coroutine
def view_log(websocket, path):
    logging.debug('Connected, remote={}, path={}'.format(websocket.remote_address, path))

    try:
        try:
            parse_result = urlparse(path)
        except Exception:
            raise ValueError('Fail to parse URL')

        # print("parse_result ", parse_result)

        page_name = parse_result.path

        # print("page_name ", page_name)
        allowed = False
        if page_name.startswith("/logview"):
            allowed = True

        if not allowed:
            raise ValueError('Forbidden')

        query = parse_qs(parse_result.query)
        # print(query)
        job_id = query and query['job_id'] and query['job_id'][0]
        log_type = query and query['lt'] and query['lt'][0]
        # print("@@@@@@@ ", job_id, log_type)
        #
        if log_type == "debug":
            fn = "run_debug.log"
        elif log_type == "info":
            fn = "run_info.log"
        elif log_type == "error":
            fn = "run_error.log"
        else:
            fn = None

        all_jobs_fld = os.path.abspath(log_folder)

        job_id_fld = os.path.join(all_jobs_fld, job_id)
        job_config_fld_path = os.path.join(job_id_fld, ".config")
        job_log_fld_path = os.path.join(job_config_fld_path, "logs")

        if not fn is None:
            job_log_fp = os.path.join(job_log_fld_path, fn)

            #
            # if not os.path.isfile(file_path):
            #     raise ValueError('Not found')
            #
            # query = parse_qs(parse_result.query)
            # tail = query and query['tail'] and query['tail'][0] == '1'

            with open(job_log_fp) as f:

                content = ''.join(deque(f, NUM_LINES))
                # content = conv.convert(content, full=False)
                yield from websocket.send(content)

                if True:
                    last_heartbeat = time.time()
                    while True:
                        content = f.read()
                        if content:
                            # content = conv.convert(content, full=False)
                            yield from websocket.send(content)
                        else:
                            yield from asyncio.sleep(1)

                        # heartbeat
                        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                            try:
                                yield from websocket.send('ping')
                                pong = yield from asyncio.wait_for(websocket.recv(), 5)
                                # print(time.time() , pong)
                                if pong != 'pong':
                                    raise Exception()
                            except Exception:
                                raise Exception('Ping error')
                            else:
                                # print("Updating heartbeat")
                                last_heartbeat = time.time()

                else:
                    yield from websocket.close()

    except ValueError as e:
        try:
            yield from websocket.send('<font color="red"><strong>{}</strong></font>'.format(e))
            yield from websocket.close()
        except Exception:
            pass

        log_close(websocket, path, e)

    except Exception as e:
        log_close(websocket, path, e)

    else:
        log_close(websocket, path)


def log_close(websocket, path, exception=None):
    message = 'Closed, remote={}, path={}'.format(websocket.remote_address, path)
    if exception is not None and str(exception) != "Ping error":
        message += ', exception={}'.format(exception)
        logging.info(message)
    else:
        message += ', exception={}'.format(exception)
        logging.debug(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--log_fld', required=True, help='Log parent folder')
    args = parser.parse_args()

    global log_folder

    # allowed_prefixes.extend(args.prefix)
    log_folder = args.log_fld
    start_server = websockets.serve(view_log, args.host, args.port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def start_log_viewer(log_fld, host, port):
    print("Inside start_log_viewer with log_fld {} host {} and port {}".format(log_fld, host, port))
    global log_folder

    # allowed_prefixes.extend(args.prefix)
    log_folder = log_fld
    asyncio.set_event_loop(asyncio.new_event_loop())
    start_server = websockets.serve(view_log, host, port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    main()
