"""Download a pinned multilingual query/passage relevance model, never a solver."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
REVISION = '1427fd652930e4ba29e8149678df786c240d8825'
FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'sentencepiece.bpe.model', 'model.safetensors', 'README.md']


def main():
    target = ROOT/'models/mmarco-mMiniLMv2-L12-H384-v1'
    target.mkdir(parents=True, exist_ok=True)
    lock = target/'download_manifest.json'
    previous = json.loads(lock.read_text(encoding='utf-8')) if lock.exists() else None
    if previous and (previous['model'], previous['revision']) != (MODEL, REVISION):
        raise ValueError('Existing reranker identity differs')
    def fetch(name):
        path = target/name
        if not path.exists():
            temp = path.with_suffix(path.suffix+'.part')
            with requests.get(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',
                              stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                with temp.open('wb') as stream:
                    for block in response.iter_content(1024*1024):
                        stream.write(block)
            temp.replace(path)
        result = {'path': name, 'bytes': path.stat().st_size,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if previous and result != next(r for r in previous['files'] if r['path'] == name):
            raise ValueError('Frozen source reranker file changed: '+name)
        print(name, result['bytes'], flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, FILES))
    result = {'model': MODEL, 'revision': REVISION, 'files': files,
        'prepared_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'source': f'https://huggingface.co/{MODEL}/tree/{REVISION}', 'license': 'Apache-2.0',
        'training': 'Upstream MMARCO, machine-translated MS MARCO; no local training or benchmark input.',
        'usage': 'Optional frozen CPU query/passage relevance ranking; no answer generation or original-source/index modification. Generator remains independently configured.'}
    if not previous:
        lock.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
