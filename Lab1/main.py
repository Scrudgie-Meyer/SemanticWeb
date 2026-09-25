"""Точка входу: послідовність виконання лабораторної роботи."""

import argparse
from rdflib import Graph

from config import OUTPUT_DIR
from graph_builder import build_graph
from graph_io import save_graph_files, print_graph_preview
from query_runner import execute_queries, execute_federated_query
from visualization import visualize

def parse_arguments() -> argparse.Namespace:
    """Читає параметри запуску з командного рядка."""
    parser = argparse.ArgumentParser(
        description="RDF-каталог автомобілів та SPARQL-запити"
    )
    parser.add_argument(
        "--federated",
        action="store_true",
        help="Виконати запит до Wikidata (потрібен інтернет)",
    )
    return parser.parse_args()


def main() -> None:
    """Виконує етапи лабораторної роботи в послідовному порядку."""
    arguments = parse_arguments()
    OUTPUT_DIR.mkdir(exist_ok=True)

    original_graph = build_graph()
    edited_path = save_graph_files(original_graph)

    # Подальші дії використовують граф із ручними змінами.
    edited_graph = Graph()
    edited_graph.parse(edited_path, format="turtle")
    visualize(edited_graph)
    execute_queries(edited_graph)
    print_graph_preview(edited_graph, edited_path)

    if arguments.federated:
        execute_federated_query(edited_graph)
    print("\nРезультати збережено в папці output.")


if __name__ == "__main__":
    main()
