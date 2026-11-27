"""Fetch the pinned multilingual retrieval encoder used by an optional selector."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
FILES = ['config.json', 'modules.json', 'sentence_bert_config.json', '1_Pooling/config.json',
         'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'sentencepiece.bpe.model', 'model.safetensors', 'README.md']


def main():
    target = ROOT/'models/multilingual-e5-small'
    target.mkdir(parents=True, exist_ok=True)
    lock = target/'download_manifest.json'
    previous = json.loads(lock.read_text(encoding='utf-8')) if lock.exists() else None
    if previous and (previous['model'], previous['revision']) != (MODEL, REVISION):
        raise ValueError('Existing encoder resource identity differs')
    def fetch(name):
        path = target/name
        if not path.exists():
            temp = path.with_suffix(path.suffix+'.part')
            path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',
                              stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                with temp.open('wb') as stream:
                    for block in response.iter_content(1024*1024):
                        stream.write(block)
            temp.replace(path)
        result = {'path': name, 'bytes': path.stat().st_size,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if previous:
            expected = next(r for r in previous['files'] if r['path'] == name)
            if result != expected:
                raise ValueError(f'Existing frozen encoder file changed: {name}')
        print(name, result['bytes'], flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, FILES))
    result = {'model': MODEL, 'revision': REVISION, 'files': files,
        'prepared_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'source': f'https://huggingface.co/{MODEL}/tree/{REVISION}',
        'usage': 'Frozen optional CPU source-window selector; no training, answer generation or original-source index rebuild. Qwen solver unchanged.'}
    if not previous:
        lock.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
