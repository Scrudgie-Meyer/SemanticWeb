"""Читання SPARQL із .rq-файлів і збереження результатів виконання."""

import socket
from rdflib import Graph
from config import (
    OUTPUT_DIR, QUERY_DIR, FEDERATED_QUERY_FILE, NETWORK_TIMEOUT_SECONDS,
)

def execute_queries(graph: Graph) -> None:
    """Виконує локальні запити й зберігає таблиці результатів."""
    query_files = sorted(
        path for path in QUERY_DIR.glob("*.rq")
        if path.name != FEDERATED_QUERY_FILE
    )
    if not query_files:
        raise FileNotFoundError(f"Не знайдено локальних .rq-запитів у {QUERY_DIR}")
    lines = [f"Кількість RDF-трійок: {len(graph)}", ""]
    for query_path in query_files:
        filename = query_path.name
        query = query_path.read_text(encoding="utf-8")
        lines.extend(["=" * 78, filename, "=" * 78])
        result = graph.query(query)
        headers = [str(variable) for variable in result.vars]
        lines.append(" | ".join(headers))
        lines.append("-" * 78)
        for row in result:
            lines.append(" | ".join(str(value) for value in row))
        lines.append("")
    (OUTPUT_DIR / "query_results.txt").write_text("\n".join(lines), encoding="utf-8")


def execute_federated_query(graph: Graph) -> None:
    """Виконує запит до Wikidata; у разі помилки зберігає попередній файл."""
    previous_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
        query = (QUERY_DIR / FEDERATED_QUERY_FILE).read_text(encoding="utf-8")
        result = graph.query(query)
        lines = [
            f"Результат запиту {FEDERATED_QUERY_FILE}",
            " | ".join(str(variable) for variable in result.vars),
        ]
        for row in result:
            lines.append(" | ".join(str(value) for value in row))

        result_path = OUTPUT_DIR / "wikidata_query_result.txt"
        result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as error:
        print(f"Запит Wikidata не виконано: {error}. Попередній результат не оновлено.")
        raise SystemExit(1) from error
    finally:
        socket.setdefaulttimeout(previous_timeout)
