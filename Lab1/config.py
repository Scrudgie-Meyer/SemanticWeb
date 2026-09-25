"""Шляхи, RDF-простори імен і параметри запуску лабораторної."""

from pathlib import Path
from rdflib import Namespace

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
QUERY_DIR = BASE_DIR / "queries"
FEDERATED_QUERY_FILE = "08_wikidata_federated.rq"
NETWORK_TIMEOUT_SECONDS = 30
PREVIEW_TRIPLE_COUNT = 12

CARS = Namespace("http://example.org/cars/")
SCHEMA = Namespace("https://schema.org/")
WD = Namespace("http://www.wikidata.org/entity/")
PREVIEW_ENTITY = CARS.ToyotaCorolla
