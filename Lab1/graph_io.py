"""Серіалізація графу та зрозумілий консольний вивід."""

from pathlib import Path
from rdflib import Graph, URIRef
from config import CARS, OUTPUT_DIR, PREVIEW_ENTITY, PREVIEW_TRIPLE_COUNT

def format_term(value: object) -> str:
    """Скорочує URI власного словника для консольного виводу."""
    text = str(value)
    if text.startswith(str(CARS)):
        return "cars:" + text[len(str(CARS)):]
    return text


def save_graph_files(original_graph: Graph) -> Path:
    """Зберігає початковий граф, не перезаписуючи файл із ручними змінами."""
    original_graph.serialize(OUTPUT_DIR / "cars_original.ttl", format="turtle")
    original_graph.serialize(
        OUTPUT_DIR / "cars.jsonld",
        format="json-ld",
        indent=2,
        ensure_ascii=False,
    )

    edited_path = OUTPUT_DIR / "cars_edited.ttl"
    if not edited_path.exists():
        original_graph.serialize(edited_path, format="turtle")
    return edited_path


def print_entity_triples(graph: Graph, entity: URIRef) -> None:
    """Виводить усі трійки, в яких вибрана сутність є суб’єктом."""
    print(f"\nТрійки для сутності {entity}:")
    for subject, predicate, obj in graph.triples((entity, None, None)):
        print(f"  {format_term(subject)} | {format_term(predicate)} | {format_term(obj)}")


def print_graph_preview(graph: Graph, source_path: Path) -> None:
    """Показує приклади трійок і властивості вибраного автомобіля."""
    print(f"Граф прочитано з {source_path.name}. Кількість трійок: {len(graph)}")
    print(f"Перші {PREVIEW_TRIPLE_COUNT} трійок:")
    for index, triple in enumerate(graph):
        if index == PREVIEW_TRIPLE_COUNT:
            break
        print("  " + " | ".join(format_term(value) for value in triple))
    print_entity_triples(graph, PREVIEW_ENTITY)
