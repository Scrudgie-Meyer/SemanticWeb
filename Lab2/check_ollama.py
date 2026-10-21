"""Verify both local models and save a real structured-output/embedding smoke test."""
import os
import numpy as np
from pipeline import BASE, Model, write_json
from schema import Extraction

def main():
    model = Model('ollama')
    embed_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'qwen3-embedding:0.6b')
    installed = {m.model for m in model.client.list().models}
    for name in [model.name, embed_model]:
        if name not in installed:
            raise RuntimeError(f'Model is missing: {name}. Run ollama pull {name}.')
    text = 'Салонний фільтр підбирають за перевіркою VIN.'
    facts = Extraction.model_validate_json(model.generate(
        'Видобудь рівно один факт SELECTED_BY з речення. Тип subject — part, '
        'тип object — criterion. Усі назви українською у називному відмінку; evidence — '
        'дослівна цитата всього речення. Речення: ' + text, structured=True)).model_dump()
    assert len(facts['facts']) == 1
    fact = facts['facts'][0]
    assert fact['relation'] == 'SELECTED_BY' and fact['evidence'] in text
    summary = model.generate('Підсумуй українською одним реченням, не більше 20 слів. '
                             'Дані: салонний фільтр підбирають за VIN; огляд передує заміні.',
                             max_tokens=128)
    assert summary.strip() and not summary.lstrip().startswith('Хм')
    vectors = model.client.embed(model=embed_model,
        input=['Слабкий потік повітря з вентиляції.', 'У салоні душно, з дефлекторів ледь дме.'],
        truncate=False).embeddings
    matrix = np.asarray(vectors)
    assert matrix.ndim == 2 and matrix.shape[0] == 2 and matrix.shape[1] > 100
    assert np.isfinite(matrix).all() and (np.linalg.norm(matrix, axis=1) > 0).all()
    result = {'success': True, 'llm': model.name, 'embedding_model': embed_model,
              'structured_output': facts, 'short_summary': summary,
              'embedding_shape': list(matrix.shape)}
    write_json(BASE / 'output' / 'ollama_check.json', result)
    print(f'Ollama OK: {model.name}; embeddings {embed_model}, shape={matrix.shape}', flush=True)

if __name__ == '__main__':
    main()
