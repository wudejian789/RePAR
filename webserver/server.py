"""Serve the local case analysis dashboard and v5 research workspace."""
from __future__ import annotations

import argparse
import functools
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

WEBROOT = Path(__file__).resolve().parent
STUDIO_PREFIX = "/api/studio/"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=WEBROOT, studio=None, **kwargs):
        self.studio = studio
        super().__init__(*args, directory=str(directory), **kwargs)

    def send_json(self, status, data):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def studio_request(self):
        if self.studio is None:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "当前服务未启用研究工作台。"})
            return
        from webserver.studio import MAX_BODY, SessionBusyError

        url = urlsplit(self.path)
        parts = url.path.removeprefix(STUDIO_PREFIX).strip("/").split("/")
        try:
            if self.command == "DELETE":
                if len(parts) != 2 or parts[0] != "sessions":
                    raise FileNotFoundError("Unknown studio resource")
                data = self.studio.delete_session(parts[1])
            elif self.command == "POST":
                if self.headers.get_content_type() != "application/json":
                    self.send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "使用 application/json 提交。"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    raise ValueError("请求正文为空。")
                if length > MAX_BODY:
                    self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "上传请求超过 16 MiB。"})
                    return
                data = self.studio.submit(parts, json.loads(self.rfile.read(length).decode("utf-8")))
            else:
                offset = int(parse_qs(url.query).get("offset", [0])[0])
                if not 0 <= offset <= 2**63 - 1:
                    raise ValueError("无效的轨迹读取位置。")
                data = self.studio.get(parts, offset)
        except FileNotFoundError:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "找不到该会话、知识库、运行记录或索引。"})
            return
        except SessionBusyError as error:
            self.send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            return
        except (ValueError, OverflowError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except OSError:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "无法读写工作台记录，请重试。"})
            return
        self.send_json(HTTPStatus.ACCEPTED if self.command == "POST" else HTTPStatus.OK, data)

    def do_POST(self):
        if urlsplit(self.path).path.startswith(STUDIO_PREFIX):
            self.studio_request()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    do_DELETE = do_POST

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith(STUDIO_PREFIX):
            self.studio_request()
            return
        if path == "/health":
            data = Path(self.directory) / "data"
            datasets = {}
            for name, relative in (("longbench", "overview.json"), ("math", "math/overview.json")):
                overview = data / relative
                if overview.is_file():
                    try:
                        saved = json.loads(overview.read_text(encoding="utf-8"))
                        datasets[name] = {"cases": len(saved["cases"])}
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
            self.send_json(HTTPStatus.OK, {"status": "ok",
                           "cases": datasets.get("longbench", {}).get("cases", 0),
                           "datasets": datasets})
            return
        if path in {"/data/overview.json", "/data/math/overview.json"} and not Path(self.translate_path(path)).is_file():
            self.send_json(HTTPStatus.OK, {"cases": [], "methods": {}, "case_count": 0})
            return
        if path in {"/studio", "/studio/libraries"}:
            self.path = "/studio.html"
        super().do_GET()

    def send_head(self):
        path = Path(self.translate_path(self.path)).resolve()
        root = Path(self.directory).resolve()
        allowed = path == root or (path.is_relative_to(root) and (path.name in ("index.html", "studio.html") or
                  path.relative_to(root).parts[0] in ("static", "data")))
        if not allowed or path.is_dir():
            if path == root:
                self.path = "/index.html"
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return None
        return super().send_head()

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        # Cytoscape injects this fixed stylesheet, so allow only its bundled hash.
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; style-src-elem 'self' 'sha256-pgvDUBa4IjFA2yuSJ2cqcyxmNYJMborsd0ORcRv9vw8='; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'")
        super().end_headers()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--studio-config", type=Path, default=Path("configs/repar_v5.json"))
    p.add_argument("--studio-root", type=Path, default=Path("artifacts/studio_v5"))
    args = p.parse_args()
    mimetypes.add_type("application/javascript", ".js")
    studio = None
    if args.studio_config.is_file():
        from webserver.studio import Studio
        from webserver.studio_examples import example_libraries
        examples = example_libraries(WEBROOT.parent)
        studio = Studio(args.studio_root, args.studio_config, examples=examples)
    with ThreadingHTTPServer((args.host, args.port), functools.partial(
            Handler, directory=WEBROOT, studio=studio)) as server:
        print(f"RePAR Case Lab: http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if studio:
                studio.workers.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
