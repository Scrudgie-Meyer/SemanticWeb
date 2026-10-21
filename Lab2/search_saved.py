"""Query a saved Ollama index without rerunning extraction or GraphRAG."""
import argparse
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pipeline import BASE, Model, Search, read_json, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--top', type=int, default=3)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive')
    out = BASE / 'output' / 'live'
    metadata = read_json(out / 'vector_store.json')
    if metadata['backend'] != 'ollama':
        parser.error('This launcher requires an Ollama index. Run run_lab.ps1 first.')
    os.environ['OLLAMA_EMBEDDING_MODEL'] = metadata['model']
    search = Search.__new__(Search)
    search.chunks, search.backend, search.out = read_json(out / 'chunks.json'), 'ollama', out
    search.model = Model('ollama')
    search.matrix = np.load(out / 'embeddings.npy', allow_pickle=False)
    if len(search.chunks) != search.matrix.shape[0]:
        raise ValueError('Vector index and chunks do not match; rerun the pipeline.')
    search.keyword_model = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b[\w-]+\b')
    search.keyword_matrix = search.keyword_model.fit_transform([c['text'] for c in search.chunks])
    result = {'query': args.query, 'keyword': search.search(args.query, args.top, keyword=True),
              'vector': search.search(args.query, args.top)}
    write_json(out / 'query_result.json', result)
    for method in ['keyword', 'vector']:
        print(f'\n{method}:')
        for hit in result[method]:
            print(f"  {hit['doc_id']} ({hit['score']:.4f}) — {hit['text'][:160]}")

if __name__ == '__main__':
    main()
