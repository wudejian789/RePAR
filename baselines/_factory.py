"""Lazy construction; no benchmark formats enter the retrieval implementations."""
def create_retriever(method, context, client=None, encoder=None, *, working_dir=None, chunk_words=240, parameters=None):
    options = dict(parameters or {})
    if method == 'BM25_RAG':
        from baselines.BM25_RAG import BM25Retriever
        from baselines.BM25_RAG.pipeline import retrieve_context
        source = BM25Retriever(context, chunk_words)
        source.retrieve = lambda query, top_k=24: retrieve_context(query, source, top_k)[1]
        return source
    if method == 'RaptorRAG':
        from baselines.RaptorRAG import RaptorRetriever
        return RaptorRetriever(context, client, encoder, chunk_words=chunk_words, **options)
    if method == 'HippoRAG2':
        from baselines.HippoRAG2 import HippoRetriever
        return HippoRetriever(context, client, encoder, working_dir=working_dir, chunk_words=chunk_words, **options)
    if method == 'HyperGraphRAG':
        from baselines.HyperGraphRAG import HyperGraphRetriever
        return HyperGraphRetriever(context, client, encoder, working_dir=working_dir, chunk_words=chunk_words, **options)
    raise ValueError(f'{method}: no main-track implementation; consult the method README')
