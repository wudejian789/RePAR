"""Default startup and persisted data work without benchmark archives or models."""
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from repar.io import write_json
from webserver import server
from webserver.studio import Studio
from webserver.studio_examples import example_libraries
from webserver.tests.test_dashboard import browser


@pytest.mark.parametrize('existing', [False, True])
def test_default_startup_never_exports_archives(tmp_path, monkeypatch, existing):
    webroot = tmp_path / 'webserver'
    webroot.mkdir()
    if existing:
        write_json(webroot / 'data/overview.json', {'cases': []})
    config = tmp_path / 'config.json'
    config.write_text('{}', encoding='utf-8')
    root = tmp_path / 'workspace'
    library = {'id': 'a' * 32, 'name': 'Saved notes', 'status': 'ready', 'created': '2026-01-01'}
    write_json(root / 'libraries' / library['id'] / 'job.json', library)
    monkeypatch.setattr(server, 'WEBROOT', webroot)
    monkeypatch.setattr(sys, 'argv', ['server', '--port', '0', '--studio-config', str(config), '--studio-root', str(root)])
    monkeypatch.setitem(sys.modules, 'webserver.build_data', SimpleNamespace(
        export=lambda: pytest.fail('default startup must not export historical archives')))

    class FakeServer:
        def __init__(self, address, handler):
            self.studio = handler.keywords['studio']

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def serve_forever(self):
            assert self.studio.library_records() == [library]
            assert self.studio.session_records() == []

    monkeypatch.setattr(server, 'ThreadingHTTPServer', FakeServer)
    server.main()
    assert not (webroot / 'data/manifest.json').exists()


def test_optional_examples_ignore_missing_or_incomplete_archive(tmp_path):
    assert example_libraries(tmp_path) == {}
    write_json(tmp_path / 'webserver/data/overview.json', {'cases': []})
    protocol = tmp_path / 'artifacts/repar_v5_20261126/seed_42/protocol.json'
    write_json(protocol, {'identity': {}})
    with pytest.warns(RuntimeWarning, match='跳过不可用'):
        assert example_libraries(tmp_path) == {}


@pytest.fixture
def empty_site(tmp_path):
    webroot = tmp_path / 'webserver'
    webroot.mkdir()
    for name in ('index.html', 'studio.html'):
        shutil.copyfile(server.WEBROOT / name, webroot / name)
    shutil.copytree(server.WEBROOT / 'static', webroot / 'static')
    root = tmp_path / 'workspace'
    studio = Studio(root, tmp_path / 'unused-config.json')
    http = ThreadingHTTPServer(('127.0.0.1', 0), partial(server.Handler, directory=webroot, studio=studio))
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}', webroot, studio
    finally:
        http.shutdown()
        http.server_close()
        worker.join()
        studio.workers.shutdown(wait=True, cancel_futures=True)


def read_url(base, path):
    with urlopen(base + path) as response:
        return json.load(response)


def test_empty_http_and_saved_data_are_loaded_without_manifest(empty_site):
    base, webroot, studio = empty_site
    assert read_url(base, '/health') == {'status': 'ok', 'cases': 0, 'datasets': {}}
    for path in ('/data/overview.json', '/data/math/overview.json'):
        assert read_url(base, path) == {'cases': [], 'methods': {}, 'case_count': 0}
    assert read_url(base, '/api/studio/libraries') == {'libraries': []}
    assert read_url(base, '/api/studio/sessions') == {'sessions': []}
    assert not (webroot / 'data').exists()

    saved = {'cases': [{'id': 'saved-case'}], 'methods': {'direct': 'Saved method'}}
    write_json(webroot / 'data/math/overview.json', saved)
    assert read_url(base, '/data/math/overview.json') == saved
    assert read_url(base, '/health')['datasets'] == {'math': {'cases': 1}}
    library = {'id': 'a' * 32, 'name': 'Saved notes', 'status': 'ready', 'created': '2026-01-01'}
    write_json(studio.root / 'libraries' / library['id'] / 'job.json', library)
    studio.submit(['sessions'], {'title': 'Saved conversation'})
    restored = Studio(studio.root, studio.config_path)
    try:
        assert restored.library_records() == [library]
        assert restored.session_records()[0]['title'] == 'Saved conversation'
    finally:
        restored.workers.shutdown(wait=True, cancel_futures=True)
    assert read_url(base, '/api/studio/libraries') == {'libraries': [library]}
    assert read_url(base, '/api/studio/sessions')['sessions'][0]['title'] == 'Saved conversation'


def test_empty_pages_render_without_archives_or_models(empty_site, browser, tmp_path):
    from playwright.sync_api import expect

    base, _, _ = empty_site
    page = browser.new_page(viewport={'width': 1440, 'height': 960})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    for suffix in ('/', '/?dataset=math'):
        page.goto(base + suffix)
        expect(page.locator('#global-status')).to_contain_text('暂无已保存的评测结果')
        expect(page.locator('.comparison-panel')).to_be_hidden()
        expect(page.locator('#export-csv')).to_be_disabled()
    page.goto(base + '/studio')
    expect(page.locator('#session-list')).to_have_text('还没有会话')
    expect(page.locator('#ask')).to_be_disabled()
    page.goto(base + '/studio/libraries')
    expect(page.locator('#library-directory')).to_have_text('没有匹配的知识库。')
    expect(page.locator('#library-summary')).to_contain_text('创建知识库后可在这里查看索引和来源文件。')
    expect(page.locator('#manager-status')).to_be_empty()
    page.screenshot(path=str(tmp_path / 'empty-library.png'), full_page=True)
    assert not errors
    page.close()
