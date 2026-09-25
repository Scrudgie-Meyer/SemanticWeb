"""Навчальні дані каталогу. Ключі словників утворюють локальні URI."""

CONTINENTS = {
    "Europe": "Європа",
    "Asia": "Азія",
    "NorthAmerica": "Північна Америка",
}

COUNTRIES = {
    "Japan": ("Японія", "Asia"),
    "Germany": ("Німеччина", "Europe"),
    "USA": ("США", "NorthAmerica"),
    "France": ("Франція", "Europe"),
}

MANUFACTURERS = {
    "Infiniti": ("Infiniti", "Japan"),
    "Nissan": ("Nissan", "Japan"),
    "Toyota": ("Toyota", "Japan"),
    "Honda": ("Honda", "Japan"),
    "BMW": ("BMW", "Germany"),
    "Tesla": ("Tesla", "USA"),
    "Ford": ("Ford", "USA"),
    "Volkswagen": ("Volkswagen", "Germany"),
    "Renault": ("Renault", "France"),
}

BODY_TYPES = {
    "SUV": "SUV",
    "Sedan": "Седан",
    "Liftback": "Ліфтбек",
    "Coupe": "Купе",
    "Hatchback": "Хетчбек",
}

ENGINE_TYPES = {
    "Petrol": "Бензиновий",
    "Diesel": "Дизельний",
    "Electric": "Електричний",
}

# Поля запису: URI, назва, рік, виробник, кузов, двигун.
CAR_RECORDS = [
    ("InfinitiQX70", "Infiniti QX70", 2014, "Infiniti", "SUV", "Petrol"),
    ("NissanQashqai", "Nissan Qashqai", 2014, "Nissan", "SUV", "Petrol"),
    ("ToyotaCorolla", "Toyota Corolla", 2013, "Toyota", "Sedan", "Petrol"),
    ("ToyotaRAV4", "Toyota RAV4", 2014, "Toyota", "SUV", "Petrol"),
    ("HondaCivic", "Honda Civic", 2012, "Honda", "Sedan", "Petrol"),
    ("BMW3Series", "BMW 3 Series", 2013, "BMW", "Sedan", "Diesel"),
    ("TeslaModelS", "Tesla Model S", 2014, "Tesla", "Liftback", "Electric"),
    ("FordMustang", "Ford Mustang", 2014, "Ford", "Coupe", "Petrol"),
    ("VolkswagenGolf", "Volkswagen Golf", 2013, "Volkswagen", "Hatchback", "Diesel"),
    ("RenaultClio", "Renault Clio", 2012, "Renault", "Hatchback", "Petrol"),
]

WIKIDATA_LINKS = {"Japan": "Q17", "Germany": "Q183"}
