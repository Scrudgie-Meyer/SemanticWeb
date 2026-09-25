"""Побудова RDF-схеми й графу з даних каталогу."""

from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD
from rdflib.namespace import DCTERMS, FOAF, OWL

from config import CARS, SCHEMA, WD
from catalog_data import (
    CONTINENTS, COUNTRIES, MANUFACTURERS, BODY_TYPES, ENGINE_TYPES, CAR_RECORDS,
    WIKIDATA_LINKS,
)

def add_named_resource(
    graph: Graph, uri: URIRef, class_uri: URIRef, name: str
) -> None:
    """Додає тип сутності та її назву до графу."""
    graph.add((uri, RDF.type, class_uri))
    graph.add((uri, SCHEMA.name, Literal(name)))


def build_graph() -> Graph:
    """Створює граф: словники, схема, навчальні дані та зовнішні зв’язки."""
    graph = Graph()
    graph.bind("cars", CARS)
    graph.bind("schema", SCHEMA)
    graph.bind("foaf", FOAF)
    graph.bind("dcterms", DCTERMS)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)

    add_schema(graph)
    add_catalog_data(graph)
    add_wikidata_links(graph)
    return graph


def add_schema(graph: Graph) -> None:
    """Описує класи та допустимі типи початку й кінця зв’язків."""
    classes = {
        CARS.Car: "Автомобіль",
        CARS.Manufacturer: "Виробник",
        CARS.Country: "Країна",
        CARS.BodyType: "Тип кузова",
        CARS.EngineType: "Тип двигуна",
        CARS.Continent: "Частина світу",
    }
    for uri, label in classes.items():
        graph.add((uri, RDF.type, RDFS.Class))
        graph.add((uri, RDFS.label, Literal(label, lang="uk")))

    properties = {
        CARS.manufacturedBy: (CARS.Car, CARS.Manufacturer, "виготовлений виробником"),
        CARS.countryOfOrigin: (CARS.Manufacturer, CARS.Country, "країна походження"),
        CARS.hasBodyType: (CARS.Car, CARS.BodyType, "має тип кузова"),
        CARS.hasEngineType: (CARS.Car, CARS.EngineType, "має тип двигуна"),
        CARS.locatedIn: (CARS.Country, CARS.Continent, "розташована у"),
    }
    for predicate, (source_class, target_class, label) in properties.items():
        graph.add((predicate, RDF.type, RDF.Property))
        graph.add((predicate, RDFS.domain, source_class))
        graph.add((predicate, RDFS.range, target_class))
        graph.add((predicate, RDFS.label, Literal(label, lang="uk")))


def add_catalog_data(graph: Graph) -> None:
    """Додає екземпляри сутностей і зв’язки між ними."""

    for key, label in CONTINENTS.items():
        add_named_resource(graph, CARS[key], CARS.Continent, label)
    for key, (label, continent) in COUNTRIES.items():
        add_named_resource(graph, CARS[key], CARS.Country, label)
        graph.add((CARS[key], CARS.locatedIn, CARS[continent]))
    for key, (label, country) in MANUFACTURERS.items():
        add_named_resource(graph, CARS[key], CARS.Manufacturer, label)
        graph.add((CARS[key], CARS.countryOfOrigin, CARS[country]))
        graph.add((CARS[key], FOAF.name, Literal(label)))
    for key, label in BODY_TYPES.items():
        add_named_resource(graph, CARS[key], CARS.BodyType, label)
    for key, label in ENGINE_TYPES.items():
        add_named_resource(graph, CARS[key], CARS.EngineType, label)
    for key, name, year, manufacturer, body_type, engine_type in CAR_RECORDS:
        uri = CARS[key]
        add_named_resource(graph, uri, CARS.Car, name)
        graph.add((uri, SCHEMA.productionDate, Literal(year, datatype=XSD.gYear)))
        graph.add((uri, CARS.manufacturedBy, CARS[manufacturer]))
        graph.add((uri, CARS.hasBodyType, CARS[body_type]))
        graph.add((uri, CARS.hasEngineType, CARS[engine_type]))


def add_wikidata_links(graph: Graph) -> None:
    """Пов’язує локальні країни з тими самими країнами у Wikidata."""
    for country_key, identifier in WIKIDATA_LINKS.items():
        country = CARS[country_key]
        graph.add((country, OWL.sameAs, WD[identifier]))
        source_uri = URIRef(f"https://www.wikidata.org/wiki/{identifier}")
        graph.add((country, DCTERMS.source, source_uri))
