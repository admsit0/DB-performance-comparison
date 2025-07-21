############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/data_generation.py
# Due Date: 22/October/2024
# Description: contains all the classes and functions neccesary for the generate data section
############################################################################################################

import csv
import random
from datetime import datetime
from faker import Faker
from tqdm import tqdm
from typing import List, Dict, Union

class UserProvider:
    """Class to generate unique users with associated data.
    Note: max 10 million users! After that a minimum probabilty (~0) of collapse
    because of no more possible uniques in some fields."""

    def __init__(self, postal_codes_file: str = 'data/cod_post_mun.csv') -> None:
        self.cities_postal_codes = self.load_postal_codes(postal_codes_file)
        self.fake = Faker('es_ES')
        self.PROVINCES = {
            '02': 'Albacete', '03': 'Alicante/Alacant', '04': 'Almería', '01': 'Araba/Álava', '33': 'Asturias', 
            '05': 'Ávila', '06': 'Badajoz', '07': 'Balears, Illes', '08': 'Barcelona', '48': 'Bizkaia', 
            '09': 'Burgos', '10': 'Cáceres', '11': 'Cádiz', '39': 'Cantabria', '12': 'Castellón/Castelló', 
            '13': 'Ciudad Real', '14': 'Córdoba', '15': 'Coruña, A', '16': 'Cuenca', '20': 'Gipuzkoa', 
            '17': 'Girona', '18': 'Granada', '19': 'Guadalajara', '21': 'Huelva', '22': 'Huesca', 
            '23': 'Jaén', '24': 'León', '25': 'Lleida', '27': 'Lugo', '28': 'Madrid', '29': 'Málaga', 
            '30': 'Murcia', '31': 'Navarra', '32': 'Ourense', '34': 'Palencia', '35': 'Palmas, Las', 
            '36': 'Pontevedra', '26': 'Rioja, La', '37': 'Salamanca', '38': 'Santa Cruz de Tenerife', 
            '40': 'Segovia', '41': 'Sevilla', '42': 'Soria', '43': 'Tarragona', '44': 'Teruel', 
            '45': 'Toledo', '46': 'Valencia/València', '47': 'Valladolid', '49': 'Zamora', 
            '50': 'Zaragoza', '52': 'Ceuta', '53': 'Melilla'
        }

        self.unique_dnis = set()
        self.generated_emails = set()
        self.generated_mobiles = set()
        self.generated_fixes = set()

        self.province_cities = {}
        for municipality_id, cities in self.cities_postal_codes.items():
            for city in cities:
                province_code = city['codigo_postal'][:2]
                if province_code not in self.province_cities:
                    self.province_cities[province_code] = []
                self.province_cities[province_code].append(city)

    @staticmethod
    def load_postal_codes(csv_file: str) -> Dict[str, List[Dict[str, str]]]:
        """Load postal codes from a CSV file."""
        postal_codes = {}
        with open(csv_file, mode='r', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                postal_codes.setdefault(row['municipio_id'], []).append({
                    'ciudad': row['municipio_nombre'],
                    'codigo_postal': row['codigo_postal']
                })
        return postal_codes

    def generate_unique_dni(self) -> str:
        """Generate a unique DNI."""
        while True:
            number_dni = f"{random.randint(0, 99999999):08d}"
            letter_dni = "TRWAGMYFPDXBNJZSQVHLCKE"[int(number_dni) % 23]
            dni = f"{number_dni}{letter_dni}"
            if dni not in self.unique_dnis:
                self.unique_dnis.add(dni)
                return dni

    def generate_unique_mobile(self) -> str:
        """Generate a unique mobile phone number."""
        while True:
            mobile_phone = self.fake.phone_number().replace(" ", "")
            if mobile_phone not in self.generated_mobiles:
                self.generated_mobiles.add(mobile_phone)
                return mobile_phone

    def generate_unique_fixed(self) -> str:
        """Generate a unique fixed phone number."""
        while True:
            fixed_phone = '9' + self.fake.phone_number().replace(" ", "")[4:]
            if fixed_phone not in self.generated_fixes:
                self.generated_fixes.add(fixed_phone)
                return fixed_phone

    def generate_unique_email(self, name: str) -> str:
        """Generate a unique email based on the user's name."""
        cleaned_name = name.lower().replace(" ", "")
        domain = self.fake.free_email_domain()
        while True:
            email = f"{cleaned_name}{random.randint(1, 9999999)}@{domain}"
            if email not in self.generated_emails:
                self.generated_emails.add(email)
                return email

    def select_city_by_province(self, province_code: str) -> Union[Dict[str, str], None]:
        """Select a random city from a given province code."""
        valid_cities = self.province_cities.get(province_code, [])
        return random.choice(valid_cities) if valid_cities else None

    def generate_user(self) -> Dict[str, Union[str, None]]:
        """Generate a user with associated data."""
        province_code = random.choice(list(self.PROVINCES.keys()))
        city_code = self.select_city_by_province(province_code)

        name = self.fake.name()
        return {
            'nombre': name,
            'dni': self.generate_unique_dni(),
            'email': self.generate_unique_email(name),
            'telefono_movil': self.generate_unique_mobile(),
            'telefono_fijo': self.generate_unique_fixed(),
            'direccion': self.fake.street_address(),
            'ciudad': city_code['ciudad'] if city_code else 'Desconocida',
            'codigo_postal': city_code['codigo_postal'] if city_code else 'Desconocida',
            'provincia': self.PROVINCES[province_code],
        }

    def generate_users(self, quantity: int) -> List[Dict[str, Union[str, None]]]:
        """Generate a list of users."""
        return [self.generate_user() for _ in tqdm(range(quantity))]


class VehicleProvider:
    """Class to generate vehicles with associated data."""

    def __init__(self, cars_example_file: str = 'data/coches.csv') -> None:
        self.fake = Faker('es_ES')
        self.CATEGORIES = [
            'urbanos', 'sedán', 'berlina', 'cupé', 'descapotable',
            'deportivo', 'todoterreno', 'monovolumen', 'SUV', 'furgoneta']
        
        self.brands_models = self.load_brands_models(cars_example_file)
        self.used_plates = set()
        
        self.consonants = 'BCDFGHJKLMNPRSTVWXYZ'
        self.letters_by_year = {
            'B': (2000, 2002),
            'C': (2002, 2004),
            'D': (2004, 2006),
            'F': (2006, 2008),
            'G': (2008, 2010),
            'H': (2010, 2014),
            'J': (2014, 2017),
            'K': (2017, 2019),
            'L': (2019, 2022),
            'M': (2022, 2023)
        }
        self.PROV_CODES = ['A', 'AB', 'AL', 'AV', 'B', 'BA', 'BI', 'BU', 'C', 'CA', 
                           'CC', 'CO', 'CR', 'CS', 'CU', 'GR', 'GU', 'H', 'HU', 
                           'J', 'L', 'LE', 'LO', 'LU', 'M', 'MA', 'MU', 'NA', 
                           'OR', 'O', 'P', 'PM', 'PO', 'S', 'SA', 'SE', 'SG', 
                           'SO', 'SS', 'T', 'TE', 'TO', 'V', 'VA', 'VI', 'Z', 'ZA']

    def load_brands_models(self, file: str) -> Dict[str, List[str]]:
        """Load car brands and models from a CSV file."""
        brands_models = {}
        with open(file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand = row['Marca']
                model = row['Modelo']
                if brand not in brands_models:
                    brands_models[brand] = []
                brands_models[brand].append(model)
        return brands_models

    def get_first_letter(self, year: int) -> str:
        """Get the first letter corresponding to the year of registration."""
        for letter, (start, end) in self.letters_by_year.items():
            if start <= year <= end:
                return letter
        return 'B'

    def generate_plate(self, year: int) -> str:
        """Generate a unique license plate with the format based on the given year."""

        limit = datetime(2000, 9, 18)
        
        if year < limit.year or (year == limit.year and datetime(year, 9, 18) < limit):
            prov = random.choice(self.PROV_CODES)
            num = random.randint(1000, 9999)
            letters = ''.join(random.choices(self.consonants, k=2))
            plate = f"{prov}-{num}-{letters}"
            if plate not in self.used_plates:
                self.used_plates.add(plate)
                return plate
            return self.generate_plate(year)
        else:
            first = self.get_first_letter(year)
            remaining = ''.join(random.choices(self.consonants, k=2))
            num = random.randint(1000, 9999)
            plate = f"{num}-{first}{remaining}"
            if plate not in self.used_plates:
                self.used_plates.add(plate)
                return plate
            return self.generate_plate(year)
        
    def generate_vehicle(self, user_dni) -> Dict[str, Union[str, None]]:
        """Generate a full vehicle with associated data."""
        brand = random.choice(list(self.brands_models.keys()))
        model = random.choice(self.brands_models[brand])
        year = random.randint(1990, 2023)
        return {
            'matricula': self.generate_plate(year),
            'numero_bastidor': self.fake.unique.vin(),
            'anno': year,
            'marca': brand,
            'modelo': model,
            'categoria': random.choice(self.CATEGORIES),
            'dni_usuario': user_dni
        }

    def generate_vehicles(self, vehicle_count: int, users) -> List[Dict[str, Union[str, None]]]:
        """Generate a list of vehicles for a given user."""
        return [self.generate_vehicle(random.choice(users)['dni']) for _ in tqdm(range(vehicle_count))]


class MasterProvider:
    """Master provider to generate users and their associated vehicles."""

    def __init__(self, post_cod_path: str = 'data/cod_post_mun.csv', cars_path: str = 'data/coches.csv') -> None:
        self.__user_provider = UserProvider(post_cod_path)
        self.__vehicle_provider = VehicleProvider(cars_path)

    def generate(self, user_quantity: int, vehicle_count: int) -> List[Dict[str, Union[Dict[str, Union[str, None]], List[Dict[str, Union[str, None]]]]]]:
        """Generate users along with their vehicles."""
        users = self.__user_provider.generate_users(user_quantity)
        vehicles = self.__vehicle_provider.generate_vehicles(vehicle_count, users)

        return users, vehicles
    
    def generate_users(self, user_quantity: int) -> List[Dict[str, Union[str, None]]]:
        """Generate a list of users."""
        return self.__user_provider.generate_users(user_quantity)
    
    def generate_vehicles(self, vehicle_count: int, users: List[Dict[str, Union[str, None]]]) -> List[Dict[str, Union[str, None]]]:
        """Generate a list of vehicles for a given user."""
        return self.__vehicle_provider.generate_vehicles(vehicle_count, users)
                                                                          
    def __str__(self) -> str:
        """Offers a string representation of the class."""
        return f"MasterProvider(user_provider={self.__user_provider}, vehicle_provider={self.__vehicle_provider})"
    

