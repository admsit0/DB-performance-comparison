############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/util.py
# Due Date: 22/October/2024
# Description: utils and helper functions for the project
############################################################################################################

import psycopg2
import sqlite3
import duckdb
from pymongo import MongoClient
import csv
import json
import os
import time
import timeit
import numpy as np
from memory_profiler import memory_usage
import sys
from typing import Dict


PK_QUERIES = {
    "psql": """SELECT dni FROM usuarios;""", # Only one PK per table
    "mongo": {"_id": 1},
    "sqlite": """PRAGMA table_info(usuarios);""",
    "duck": """PRAGMA table_info('usuarios');"""
}

JOIN_QUERIES = {
    "psql": """SELECT u.nombre, u.dni, v.matricula, v.modelo
               FROM usuarios u
               INNER JOIN vehiculos v ON u.dni = v.dni_usuario;""",
    "mongo": [
        {
            '$lookup': {
                'from': 'vehicles',
                'localField': 'dni',
                'foreignField': 'dni_usuario',
                'as': 'vehiculos_info'
            }
        }
    ],
    "sqlite": """SELECT u.nombre, u.dni, v.matricula, v.modelo
                 FROM usuarios u
                 INNER JOIN vehiculos v ON u.dni = v.dni_usuario;""",
    "duck": """SELECT u.nombre, u.dni, v.matricula, v.modelo
               FROM usuarios u
               INNER JOIN vehiculos v ON u.dni = v.dni_usuario;"""
}


SELECT_ALL_QUERIES = {
    "psql": 'SELECT * FROM usuarios',
    "mongo": {"_id": 0},
    "sqlite": 'SELECT * FROM usuarios',
    "duck": 'SELECT * FROM usuarios'
}


def get_connection(db_type: str, params: Dict[str, str]):
    """
    Returns a connection to the specified database type.
    
    Args:
        db_type (str): The type of database ('postgres', 'sqlite', 'mongo', 'duckdb').
        params (dict): Connection parameters for the database.
        
    Returns:
        conn: Connection object or client for the specified database.
        
    Raises:
        ValueError: If the specified database type is unsupported.
    """
    if db_type == 'psql':
        conn = psycopg2.connect(**params)
        return conn

    elif db_type == 'sqlite':
        conn = sqlite3.connect(params['db'])
        return conn

    elif db_type == 'mongo':
        client = MongoClient(params['uri'])
        return client
    
    elif db_type == 'duck':
        conn = duckdb.connect(params['db'])
        return conn

    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    

def merge_data(users, vehicles):
    """Function used to merge users and vehicles data into one."""
    result = []

    # Create a dictionary for quick access to users by their DNI
    user_dict = {user["dni"]: user for user in users}

    # Dictionary to associate users with a list of vehicles
    users_with_vehicles = {}

    for vehicle in vehicles:
        user_dni = vehicle["dni_usuario"]

        if user_dni in user_dict:
            user = user_dict[user_dni]

            if user_dni not in users_with_vehicles:
                users_with_vehicles[user_dni] = {
                    "nombre_usuario": user["nombre"],
                    "dni_usuario": user["dni"],
                    "email_usuario": user["email"],
                    "telefono_movil_usuario": user["telefono_movil"],
                    "telefono_fijo_usuario": user["telefono_fijo"],
                    "direccion_usuario": user["direccion"],
                    "ciudad_usuario": user["ciudad"],
                    "codigo_postal_usuario": user["codigo_postal"],
                    "provincia_usuario": user["provincia"],
                    "coches": []  # List of cars for this user
                }

            # Add the car to the user's car list
            users_with_vehicles[user_dni]["coches"].append({
                "matricula": vehicle["matricula"],
                "numero_bastidor": vehicle["numero_bastidor"],
                "anno": vehicle["anno"],
                "marca": vehicle["marca"],
                "modelo": vehicle["modelo"],
                "categoria": vehicle["categoria"]
            })
 
    return list(users_with_vehicles.values())


def create_csv(names, data):
    """Create CSV files from provided data."""

    def __users_to_csv(users, out="out/users.csv"):
        """Write user data to a CSV file."""
        fields = ['nombre', 'dni', 'email', 'telefono_movil', 'telefono_fijo', 'direccion', 'ciudad', 'codigo_postal', 'provincia']
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(out), exist_ok=True)
        
        with open(out, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for user in users:
                writer.writerow(user)

    def __vehicles_to_csv(vehicles, out="out/vehicles.csv"):
        """Write vehicle data to a CSV file."""
        fields = ['matricula', 'numero_bastidor', 'anno', 'marca', 'modelo', 'categoria', 'dni_usuario']

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(out), exist_ok=True)

        with open(out, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            
            for vehicle in vehicles:
                writer.writerow(vehicle)

    __users_to_csv(data[0], names[0])
    __vehicles_to_csv(data[1], names[1])


def create_json(names, data):
    """Create JSON files from provided data."""

    def __data_to_json(data, out="out/output.json"):
        """Write data to a JSON file."""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as file:
            json.dump(data, file, indent=4)

    for i in range(len(data)):
        __data_to_json(data[i], names[i])

    
def time_function(function, op_type: str, is_partial: bool = True):
    """Measure and return the execution time of a given function."""
    
    t1 = time.perf_counter(), time.process_time()
    function()  # Call the function to profile
    t2 = time.perf_counter(), time.process_time()

    # Access the first positional argument, which is the 'method'
    try:
        method = function.args[0]
    except AttributeError:
        method = ''

    # Conditional within the f-string to show 'method' only for insert operations
    method_text = method if op_type == 'insert' else ''

    # Change the text dynamically based on op_type
    operation_text = {
        "insert": "with insert method",
        "select": "with select operation",
        "update": "with update operation",
        "data generation": "in data generation"
    }.get(op_type, "with operation")  # Default value: "with operation"

    # Print execution time
    if is_partial:
        print(f'{function.func.__name__} {operation_text}: {method_text}')
    else:
        print(f'{function.__name__} {operation_text}: {method_text}')
    
    real_time = t2[0] - t1[0]
    cpu_time = t2[1] - t1[1]
    print(f'\tReal time: {real_time:.5f} seconds')
    print(f'\tCPU time: {cpu_time:.5f} seconds\n\n')

    return {'real_time': real_time, 'cpu_time': cpu_time}


def time_this(function, iterations, op_type, is_partial=True):
    """Measure and return the average execution time of a function over specified iterations."""
    
    # Measure the total execution time
    total_time = timeit.timeit(function, number=iterations, globals=globals())

    # Access the first positional argument, which is the 'method'
    try:
        method = function.args[0]
    except AttributeError:
        method = ''
    
    # Conditional within the f-string to show 'method' only for insert operations
    method_text = method if op_type == 'insert' else ''

    # Change the text dynamically based on op_type
    operation_text = {
        "insert": "with insert method",
        "select": "with select operation",
        "update": "with update operation",
        "data generation": "in data generation"
    }.get(op_type, "with operation")  # Default value: "with operation"

    # Print the average time
    if is_partial:
        print(f'{function.func.__name__} {operation_text}: {method_text} --> Average time in {iterations} is {total_time / iterations:.2f} seconds')
    else:
        print(f'{function.__name__} {operation_text}: {method_text} --> Average time in {iterations} is {total_time / iterations:.2f} seconds')
    
    return {'average_time': total_time / iterations}

def measure_times(functions, iterations, op_type='data generation', is_partial=True, both=True):
    """Measure execution times of multiple functions for a specified number of iterations.
    The both parameter is needingly added because of the existance of objects that cannot be 
    executed twice (db inserts need to be deleted before being inserted again, which is not possible
    with the current implementation of the functions)."""

    results = []  # Store results for each function
    for function in functions:
        time_results = time_function(function, op_type, is_partial=is_partial)
        if both: 
            average_results = time_this(function, iterations, op_type, is_partial=is_partial)
            result = {**time_results, **average_results} # Combine results and append to the list
            results.append(result)

        else: results.append(time_results)

    return results  # Return a list of dictionaries with results



def measure_memory_generation(data):
    """Function to measure memory usage of data generation."""

    memory, _ = memory_usage((lambda: data, ), max_usage=True, retval=True)
    data_size = sys.getsizeof(data)
    print(f'Max memory usage: {memory/(1024):.2f} GB')
    print(f'Memory used by data: {data_size/(1024*1024):.2f} MB')
    return memory, data_size
