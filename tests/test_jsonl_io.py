import json

from repar.io import append_jsonl, read_jsonl


def test_literal_unicode_line_separators_do_not_split_jsonl_records(tmp_path):
    path=tmp_path/"observations.jsonl"
    values=[{"id":"a","source":"alpha\u2028beta\u2029gamma\u0085delta"},
            {"id":"b","source":"second\nline"}]
    for value in values:append_jsonl(path,value)
    assert "\\u2028" not in path.read_text(encoding="utf-8")
    assert read_jsonl(path)==values
    path.write_bytes(path.read_bytes().replace(b"\r\n",b"\n").replace(b"\n",b"\r\n"))
    assert read_jsonl(path)==values
