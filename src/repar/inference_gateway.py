"""Loopback-only slot scheduler for identical local model replicas.

No retrieval, prompt rewriting, answer parsing or dataset knowledge lives here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import requests


class Gateway:
    def __init__(self, endpoints, slots, directory, timeout=600, attempts=2):
        self.endpoints = endpoints
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout, self.attempts = timeout, attempts
        self.leases = queue.Queue()
        for _ in range(slots):
            for endpoint in endpoints:
                self.leases.put(endpoint.rstrip('/'))
        self.lock = threading.Lock()
        self.completed = self.failed = 0
        self.active = {endpoint: 0 for endpoint in endpoints}

    def event(self, record):
        with self.lock:
            with (self.directory/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')

    def handle(self, route, body, case_id):
        request_id = uuid.uuid4().hex
        queued = time.perf_counter()
        queue_seconds = service_seconds = 0.0
        last = None
        for attempt in range(1, self.attempts+1):
            endpoint = self.leases.get(timeout=self.timeout)
            queue_seconds += time.perf_counter()-queued
            with self.lock:
                self.active[endpoint] += 1
            started = time.perf_counter()
            status, result = 502, None
            try:
                response = requests.post(endpoint+route, json=body, timeout=(10,self.timeout))
                status = response.status_code
                result = response.json()
                if status >= 500:
                    raise RuntimeError(f'Backend {status}: {result}')
                service_seconds += time.perf_counter()-started
                with self.lock:
                    self.completed += int(status < 400)
                    self.failed += int(status >= 400)
                return status, result, {'X-Repar-Endpoint':endpoint,
                    'X-Repar-Request-ID':request_id,
                    'X-Repar-Queue-Seconds':str(queue_seconds),
                    'X-Repar-Service-Seconds':str(service_seconds),
                    'X-Repar-Attempts':str(attempt)}
            except Exception as exc:
                last = exc
                result = {'error': {'type':type(exc).__name__, 'message':str(exc)}}
            finally:
                seconds = time.perf_counter()-started
                event = {'request_id':request_id, 'attempt':attempt, 'case_id':case_id,
                    'route':route, 'endpoint':endpoint, 'status':status,
                    'queue_seconds':queue_seconds, 'service_seconds':seconds,
                    'time_utc':datetime.now(timezone.utc).isoformat()}
                destination = self.directory/'calls'/request_id[:2]/f'{request_id}_{attempt}.json'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({**event,'request':body,'response':result},
                    ensure_ascii=False), encoding='utf-8')
                self.event(event)
                with self.lock:
                    self.active[endpoint] -= 1
                self.leases.put(endpoint)
            queued = time.perf_counter()
        with self.lock:
            self.failed += 1
        return 502, {'error': {'message':str(last), 'request_id':request_id}}, {}


def serve(gateway, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, result, headers=None):
            data = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            for key,value in (headers or {}).items():
                self.send_header(key,value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path not in ('/health','/stats'):
                return self.send(404, {'error':'Unknown endpoint'})
            if self.path == '/health':
                try:
                    statuses = {endpoint:requests.get(endpoint+'/health',timeout=5).status_code
                                for endpoint in gateway.endpoints}
                    return self.send(200 if all(x==200 for x in statuses.values()) else 503,
                                     {'backends':statuses})
                except Exception as exc:
                    return self.send(503, {'error':str(exc)})
            with gateway.lock:
                stats = {'completed':gateway.completed,'failed':gateway.failed,
                         'active':dict(gateway.active),'available_slots':gateway.leases.qsize()}
            self.send(200, stats)

        def do_POST(self):
            if self.path not in ('/v1/chat/completions','/tokenize','/detokenize'):
                return self.send(404, {'error':'Unknown route'})
            try:
                size = int(self.headers.get('Content-Length',0))
                if size <= 0 or size > 32*1024*1024:
                    return self.send(400, {'error':'Invalid request length'})
                body = json.loads(self.rfile.read(size))
                if body.get('stream'):
                    return self.send(400, {'error':'This audited gateway requires stream=false'})
                status, result, headers = gateway.handle(self.path,body,self.headers.get('X-Repar-Case',''))
                self.send(status,result,headers)
            except (BrokenPipeError,ConnectionResetError):
                pass
            except Exception as exc:
                self.send(502, {'error':str(exc),'type':type(exc).__name__})

    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads = True
    print(f'Gateway ready on 127.0.0.1:{port}',flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoints',nargs='+',required=True)
    parser.add_argument('--slots',type=int,required=True)
    parser.add_argument('--port',type=int,default=9094)
    parser.add_argument('--output',required=True)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--attempts',type=int,default=2)
    args = parser.parse_args()
    serve(Gateway(args.endpoints,args.slots,args.output,args.timeout,args.attempts),args.port)


if __name__ == '__main__':
    main()
