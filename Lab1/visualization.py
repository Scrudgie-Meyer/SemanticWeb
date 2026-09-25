"""Збереження схеми RDF-графу як PNG."""

import matplotlib

# Для створення файлів не потрібне графічне вікно.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, RDF

from config import CARS, SCHEMA, OUTPUT_DIR
from graph_io import format_term

def visualize(graph: Graph) -> None:
    """Зберігає дві панелі графу: походження та характеристики автомобілів."""
    panels = [
        (
            "Походження: manufacturedBy → countryOfOrigin → locatedIn",
            [
                (CARS.Car, "Автомобілі", "#dbeafe"),
                (CARS.Manufacturer, "Виробники", "#fef3c7"),
                (CARS.Country, "Країни", "#dcfce7"),
                (CARS.Continent, "Частини світу", "#ffedd5"),
            ],
            [CARS.manufacturedBy, CARS.countryOfOrigin, CARS.locatedIn],
        ),
        (
            "Характеристики: hasBodyType ← автомобіль → hasEngineType",
            [
                (CARS.BodyType, "Типи кузова", "#ede9fe"),
                (CARS.Car, "Автомобілі", "#dbeafe"),
                (CARS.EngineType, "Типи двигуна", "#fee2e2"),
            ],
            [CARS.hasBodyType, CARS.hasEngineType],
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(18, 16))
    fig.suptitle("RDF-граф предметної області «Автомобілі»", fontsize=21, y=0.985)
    for ax, (title, groups, predicates) in zip(axes, panels):
        diagram = nx.DiGraph()
        positions = {}
        labels = {}
        colors = {}
        for column, (class_uri, heading, color) in enumerate(groups):
            nodes = sorted(graph.subjects(RDF.type, class_uri), key=str)
            for index, node in enumerate(nodes):
                positions[node] = (column * 3, (len(nodes) - 1) / 2 - index)
                labels[node] = str(graph.value(node, SCHEMA.name) or format_term(node))
                colors[node] = color
                diagram.add_node(node)
            ax.text(column * 3, 5.6, heading, ha="center", fontsize=12, weight="bold")
        for predicate in predicates:
            relationships = sorted(
                graph.subject_objects(predicate),
                key=lambda pair: tuple(map(str, pair)),
            )
            for source, target in relationships:
                if source in positions and target in positions:
                    diagram.add_edge(source, target)
        nx.draw_networkx_edges(
            diagram, positions, ax=ax, arrows=True, arrowsize=16,
            node_size=1900, edge_color="#94a3b8", width=1.2,
        )
        for node, (x, y) in positions.items():
            ax.text(
                x, y, labels[node], ha="center", va="center", fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": colors[node],
                    "edgecolor": "#64748b",
                },
            )
        ax.set_title(title, fontsize=12, pad=14, color="#475569")
        ax.set_xlim(-1.0, (len(groups) - 1) * 3 + 1.0)
        ax.set_ylim(-5.25, 6.2)
        ax.axis("off")
    fig.text(
        0.5, 0.012,
        "Показано зв’язки між екземплярами. Автомобілі повторено на обох панелях; "
        "літерали та схема класів опущені.",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=3)
    fig.savefig(OUTPUT_DIR / "cars_graph.png", dpi=160, facecolor="white")
    plt.close(fig)
