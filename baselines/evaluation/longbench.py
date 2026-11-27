"""Official LongBench RAG prompt, shared final decoding and evidence envelope."""
from pathlib import Path
from repar.evaluation import extract_answer, qa_text
from baselines._pipeline import retrieve_context


def prompt(item, context):
    template = (Path(__file__).parent / '0shot_rag.txt').read_text(encoding='utf-8')
    replacements = {'$DOC$': context.strip(), '$Q$': item['question'].strip()}
    replacements.update({'$C_' + c + '$': item['choice_' + c].strip() for c in 'ABCD'})
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def answer(item, source, client, config):
    context, records = retrieve_context(qa_text(item), source, config['retrieval_top_k'])
    if not records:
        raise RuntimeError('No evidence retrieved; do not silently count an ungrounded answer as baseline success')
    response = client.complete([{'role': 'user', 'content': prompt(item, context)}],
        config['answer_max_tokens'], tag='baseline_longbench_answer')
    return {'response': response, 'pred': extract_answer(response), 'evidence_ids': [r['id'] for r in records],
            'evidence': records, 'index_stats': source.stats}
