############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/testing.py
# Due Date: 22/October/2024
# Description: various testing functions for the project, not organized in a specific pipeline
############################################################################################################

from typing import Dict, List, Union
from modules.db_operations import process_individual_records,get_record_keys,update_records_close_connection,update_records_open_connection
from modules.util import get_connection, merge_data, create_csv, create_json, measure_times, measure_memory_generation, time_function
from modules.connection import *
from modules.data_generation import MasterProvider
from functools import partial
import pandas as pd
import numpy as np
from numpy import mean
import seaborn as sns
import matplotlib.pyplot as plt
from modules.db_operations import delete_data, insert_data, select_all_records, update_record
from pymemcache.client import base
from modules.cache_implementation import SimpleCache
import pandas as pd
from typing import List, Dict, Union
from functools import partial

def test_data_generation(gen_names, data_sizes, provider: MasterProvider, iters: int = 5, out_path = 'out') -> pd.DataFrame:
    """Function to run tests for data generation functions and return organized results."""

    # Crear un diccionario para almacenar los resultados
    results = {
        'data_size': [],
        'step': [],
        'time': [],
        'memory': []
    }

    for size in data_sizes:
        print(f'\n\nRunning tests with data size: {size}\n{"----" * 10}')

        print('Generating users...')
        us = provider.generate_users(size)
        time_users = measure_times([lambda: provider.generate_users(size)], iterations=iters, is_partial=False)
        results['data_size'].append(size)
        results['step'].append('Generate Users')
        results['time'].append(time_users[0])
        results['memory'].append(measure_memory_generation(us))


        print('Generating vehicles...')
        veh = provider.generate_vehicles(size * 2, us)
        time_vehicles = measure_times([lambda: provider.generate_vehicles(size * 2, us)], iterations=iters, is_partial=False)
        results['data_size'].append(size)
        results['step'].append('Generate Vehicles')
        results['time'].append(time_vehicles[0])
        results['memory'].append(measure_memory_generation(veh))


        print('Merging users and vehicles...')
        mix = merge_data(us, veh)
        time_merge = measure_times([lambda: merge_data(us, veh)], iterations=iters, is_partial=False)
        results['data_size'].append(size)
        results['step'].append('Merge Data')
        results['time'].append(time_merge[0])
        results['memory'].append(measure_memory_generation(mix))

        functions_to_test = [
            partial(merge_data, us, veh),
            partial(create_csv, [f'{out_path}/csv/users.csv', f'{out_path}/csv/vehicles.csv'], [us, veh]),
            partial(create_json, [f'{out_path}/json/users.json', f'{out_path}/json/vehicles.json', f'{out_path}/json/mix.json'], [us, veh, mix]),
        ]

        for gen_fun, name in zip(functions_to_test, gen_names):
            print(f'Running tests in: {name}')

            time_function = measure_times([gen_fun], iterations=iters)
            results['data_size'].append(size)
            results['step'].append(name)
            results['time'].append(time_function[0])
            results['memory'].append(None)

            print('----' * 10)

    print('---------' * 10)

    df = pd.DataFrame(results)
    df['real_time'] = df['time'].apply(lambda x: x['real_time'])
    df['cpu_time'] = df['time'].apply(lambda x: x['cpu_time'])
    df['average_time'] = df['time'].apply(lambda x: x['average_time'])
    df.drop(columns=['time'], inplace=True)
    df['memory_used'] = df['memory'].apply(lambda x: x[0] if x is not None else 0)

    return df



def plot_test_data_generation(df: pd.DataFrame):
    """Function to plot the results of the test_data_generation function."""
    
    # Define the steps for the figure
    steps = ['Generate Users', 'Generate Vehicles', 'Merge Data']
    
    # Create a single subplot for Generate Users, Generate Vehicles, and Merge Data
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))  
    time_metrics = ['real_time', 'cpu_time', 'average_time']
    memory_metric = 'memory_used'

    # Plot execution time and memory for the steps
    for i, step in enumerate(steps):
        step_data = df[df['step'] == step]
        
        if not step_data.empty:
            # Plot each time metric
            for metric in time_metrics:
                sns.lineplot(data=step_data, x='data_size', y=metric, ax=axes[0, i], label=metric.replace('_', ' ').title())
            axes[0, i].set_title(f'{step} - Execution Time')
            axes[0, i].set_xlabel('Data Size')
            axes[0, i].set_ylabel('Time (seconds)')
            axes[0, i].legend()
            
            # Plot memory used
            sns.lineplot(data=step_data, x='data_size', y=memory_metric, ax=axes[1, i], label='Memory Used')
            axes[1, i].set_title(f'{step} - Memory Usage')
            axes[1, i].set_xlabel('Data Size')
            axes[1, i].set_ylabel('Memory (MB)')
            axes[1, i].legend()

    plt.tight_layout()
    plt.show()

    return fig


def test_db_operations(data_sizes, provider: MasterProvider, iters: int = 3, databases : List = ['psql','sqlite', 'mongo', 'duck']) -> pd.DataFrame:
    """Function to run tests for database operations and return organized results."""

    def __test_delete_operations(results: pd.DataFrame, data_size: int, iterations:int = 3, databases : List = ['psql','sqlite', 'mongo', 'duck']):
        """Test delete operations for various databases."""
        for db_name in databases:
            print(f'Deleting data from {db_name}...')
            delete_fun = partial(delete_data, db_name, 
                                PSQL_PARAMS if db_name == 'psql' else 
                                SQLITE_PARAMS if db_name == 'sqlite' else 
                                MONGO_PARAMS if db_name == 'mongo' else 
                                DUCK_PARAMS, ['usuarios', 'vehiculos'])
            time_delete = measure_times([delete_fun], iterations=iterations)
            results['data_size'].append(data_size)
            results['op_type'].append(f'delete')
            results['db_name'].append(db_name)
            results['time'].append(time_delete[0])  # Save the first result time
        return results


    def __test_insert_operations(results: pd.DataFrame, data_size: int, iterations:int = 3, databases : List = ['psql','sqlite', 'mongo', 'duck']): # Inserts has to be one iter
        """Test insert operations for various databases."""
        insert_methods = ['one', 'at_once', 'chunk']  # Default chunk size is 1000
        for method in insert_methods:
            for db_name in databases:
                for use_index in [False, True]:  # Test with and without index
                    
                    ts_real = []
                    ts_cpu = []
                    ts_avg = []

                    for _ in range(iterations):
                        # Avoid duplicate key errors
                        delete_data(db_name,
                                    PSQL_PARAMS if db_name == 'psql' else 
                                    SQLITE_PARAMS if db_name == 'sqlite' else 
                                    MONGO_PARAMS if db_name == 'mongo' else 
                                    DUCK_PARAMS, ['usuarios', 'vehiculos'])

                        print(f'\nInserting data into {db_name} using method {method} (use_index={use_index})...')
                        insert_fun = partial(insert_data, db_name,
                                            PSQL_PARAMS if db_name == 'psql' else  
                                            SQLITE_PARAMS if db_name == 'sqlite' else 
                                            MONGO_PARAMS if db_name == 'mongo' else 
                                            DUCK_PARAMS,
                                            'out/csv/users.csv' if db_name != 'mongo' else 'out/json/users.json',
                                            'usuarios', method=method, use_index=use_index)

                        time_insert = measure_times([insert_fun], iterations=1, both=False)

                        delete_data(db_name,
                                    PSQL_PARAMS if db_name == 'psql' else 
                                    SQLITE_PARAMS if db_name == 'sqlite' else 
                                    MONGO_PARAMS if db_name == 'mongo' else 
                                    DUCK_PARAMS, ['usuarios', 'vehiculos'])
                    

                        # Verifica si el diccionario contiene las claves esperadas antes de acceder
                        if time_insert and 'cpu_time' in time_insert[0] and 'real_time' in time_insert[0]:
                            ts_cpu.append(time_insert[0]['cpu_time'])
                            ts_real.append(time_insert[0]['real_time'])
                        else:
                            print(f"Warning: Missing time data for {db_name} with method {method} (use_index={use_index})")
                            ts_cpu.append(None)  # o puedes manejar estos casos de otra manera
                            ts_real.append(None)
                            ts_avg.append(None)

                    # Solo agregamos el promedio si la lista no está vacía
                    results['data_size'].append(data_size)
                    results['op_type'].append('insert')
                    results['db_name'].append(db_name)
                    results['time'].append({
                        'cpu_time': mean([t for t in ts_cpu if t is not None]) if ts_cpu else None,
                        'real_time': mean([t for t in ts_real if t is not None]) if ts_real else None,
                        'average_time': mean([t for t in ts_avg if t is not None]) if ts_avg else None
                    })
        return results



    def __test_select_operations(results: pd.DataFrame, data_size: int, iterations:int = 3, databases : List = ['psql','sqlite', 'mongo', 'duck']):
        """Test select operations for various databases."""
        for db_name in databases:
            print(f'Selecting data from {db_name}...')
            select_fun = partial(select_all_records, db_name, 
                                PSQL_PARAMS if db_name == 'psql' else 
                                SQLITE_PARAMS if db_name == 'sqlite' else 
                                MONGO_PARAMS if db_name == 'mongo' else 
                                DUCK_PARAMS, 'usuarios')
            time_select = measure_times([select_fun], iterations=iterations)
            results['data_size'].append(data_size)
            results['op_type'].append('select')
            results['db_name'].append(db_name)
            results['time'].append(time_select[0])  # Save the first result time
        return results


    def __test_update_operations(results: pd.DataFrame, data_size: int, iterations:int = 3, databases : List = ['psql','sqlite', 'mongo', 'duck']):
        """Test update operations for various databases."""
        for db_name in databases:
            print(f'Updating data in {db_name}...')
            update_fun = partial(update_record, db_name, 
                                PSQL_PARAMS if db_name == 'psql' else 
                                SQLITE_PARAMS if db_name == 'sqlite' else 
                                MONGO_PARAMS if db_name == 'mongo' else 
                                DUCK_PARAMS, 'usuarios', 'nombre', 'Adamsito', 
                                "provincia = 'Madrid'" if db_name != 'mongo' else {'provincia': 'Madrid'})
            time_update = measure_times([update_fun], iterations=iterations)
            results['data_size'].append(data_size)
            results['op_type'].append('update')
            results['db_name'].append(db_name)
            results['time'].append(time_update[0])  # Save the first result time
        return results


    # Create a dictionary to store the results
    results = {
        'data_size': [],
        'op_type': [],
        'db_name': [],
        'time': []
    }

    for size in data_sizes:
        print(f'\n\nRunning tests with data size: {size}\n{"----" * 10}')

        print('\n\nStep 1: Generating users and vehicles...')
        us, veh = provider.generate(size, int(size * 2))  # Generate users and vehicles
        create_csv(['out/csv/users.csv', 'out/csv/vehicles.csv'], [us, veh])
        create_json(['out/json/users.json', 'out/json/vehicles.json'], [us, veh])

        print('\n\nStep 2: Testing delete operations')
        __test_delete_operations(results, data_size=size, iterations=iters, databases=databases)

        print('\n\nStep 3: Testing insert operations')
        __test_insert_operations(results, data_size=size, iterations=iters, databases=databases)

        print('\n\nStep 4: Testing select operations')
        __test_select_operations(results, data_size=size, iterations=iters, databases=databases)

        print('\n\nStep 5: Testing update operations')
        __test_update_operations(results, data_size=size, iterations=iters, databases=databases)

    print('---------' * 10)


    # Convertir los resultados en DataFrame
    df = pd.DataFrame(results)

    # Extraer los tiempos reales y de CPU
    df['real_time'] = df['time'].apply(lambda x: x['real_time'] if isinstance(x, dict) and 'real_time' in x else None)
    df['cpu_time'] = df['time'].apply(lambda x: x['cpu_time'] if isinstance(x, dict) and 'cpu_time' in x else None)
    df['average_time'] = df['time'].apply(lambda x: x['average_time'] if isinstance(x, dict) and 'average_time' in x else None)

    # Quitar la columna 'time' porque ya hemos extraído los tiempos importantes
    df.drop(columns=['time'], inplace=True)

    return df


def plot_test_db_operations(df):
    """Function to plot the results of the test_db_operations function."""

    db_names = df['db_name'].unique()
    op_types = df['op_type'].unique()
    num_rows = len(db_names)
    num_cols = len(op_types)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12), sharex=True, sharey=False)

    for i, db in enumerate(db_names):
        for j, op in enumerate(op_types):
            ax = axes[i, j]  # Seleccionar el subplot correspondiente
            filtered_data = df[(df['db_name'] == db) & (df['op_type'] == op)]
            
            sns.lineplot(x='data_size', y='real_time', data=filtered_data, ax=ax, label='Real Time', marker="o", color="blue",errorbar=None)
            sns.lineplot(x='data_size', y='cpu_time', data=filtered_data, ax=ax, label='CPU Time', marker="o", color="red",errorbar=None)
            if op != 'insert': sns.lineplot(x='data_size', y='average_time', data=filtered_data, ax=ax, label='Average Time', marker="o", color="green",errorbar=None)

            if i == 0:
                ax.set_title(f'Operation: {op.capitalize()}', fontsize=14)
            if j == 0:
                ax.set_ylabel(f'{db.capitalize()} (Seconds)', fontsize=12)
            if i == num_rows - 1:
                ax.set_xlabel('Data Size', fontsize=12)

            ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
    return fig


def run_tests_cache(db_names: List[str], params: List[Dict[str, str]], sizes: List[int],
                    provider: MasterProvider, cache: Union[SimpleCache | base.Client], 
                    queries: Dict[str, str], op_type: str = 'select', method: str = 'one_each', index=False) -> pd.DataFrame:
    """
    Executes tests with different data sizes and measures time and memory.
    The queries are passed in a list and executed in each database.
    
    Args:
        db_names (list): Names of the databases (['postgres', 'sqlite', 'mongo', etc.]).
        params (list): Connection parameters for each database.
        sizes (list): Data sizes to test.
        provider: Data provider that generates users and vehicles.
        cache (Cache): Cache object.
        queries (list): List of queries to execute in each database.
        op_type (str): Type of operation to measure ('select', 'insert', 'update').
        method (str): Method to process individual records ('one_each' or 'at_once').

    Returns:
        pd.DataFrame: DataFrame containing the results of the tests.
    """

    results = {
        'data_size': [],
        'db_name': [],
        'op_type': [],
        'method': [],
        'time': [],
        'memory': [],
        'cache_used': []  # Nueva columna para identificar el uso de caché
    }

    for size in sizes:
        print('\n\nStep 1: Generating users and vehicles...')
        us, veh = provider.generate(size, int(size * 2))  # Generate users and vehicles
        create_csv(['out/csv/users.csv', 'out/csv/vehicles.csv'], [us, veh])
        create_json(['out/json/users.json', 'out/json/vehicles.json'], [us, veh])

        print(f'\n\nStep 2: Inserting in the db with index = {index}')
        for name, param in zip(db_names, params):
            conn = get_connection(name, param)
            delete_data(name, param, ['usuarios', 'vehiculos'])
            insert_data(name, param, 'out/csv/users.csv', 'usuarios', method='at_once', use_index = index)
            insert_data(name, param, 'out/csv/vehicles.csv', 'vehiculos', method='at_once', use_index = False)
            conn.close()

        print(f'\n\nRunning tests with data size: {size}\n{"----" * 10}')
    
        for name, param in zip(db_names, params):
            print('---------' * 10)
            print(f'Running tests in: {name}')
            conn = get_connection(name, param)
            query = queries[name]

            try:
                print(f"Running tests using method: {method}\n")

                records = get_record_keys(name, param, 'usuarios')
                
                # Running tests without cache
                print(f"Running tests without cache\n")
                time_without_cache = time_function(
                    partial(process_individual_records, name, param, records, query, cache, method=method), 
                    op_type=op_type
                )

                # Agregar resultados sin caché
                results['data_size'].append(size)
                results['db_name'].append(name)
                results['op_type'].append(op_type)
                results['method'].append(method)
                results['time'].append(time_without_cache)
                results['memory'].append(None)  # Ajustar según la medición de memoria
                results['cache_used'].append('No')  # Indicar que no se utilizó caché

                # Running tests with cache
                print(f"Running tests with cache\n")
                time_with_cache = time_function(
                    partial(process_individual_records, name, param, records, query, cache, method=method), 
                    op_type=op_type
                )

                # Agregar resultados con caché
                results['data_size'].append(size)
                results['db_name'].append(name)
                results['op_type'].append(op_type)
                results['method'].append(method)
                results['time'].append(time_with_cache)
                results['memory'].append(None)  # Ajustar según la medición de memoria
                results['cache_used'].append('Yes')  # Indicar que se utilizó caché

            finally:
                conn.close()
            try: cache.clear()
            except: pass # Custom accepts it, memcached does not  

        print('---------' * 10)

    df = pd.DataFrame(results)


    df['real_time'] = df['time'].apply(lambda x: x.get('real_time', None) if isinstance(x, dict) else None)
    df['cpu_time'] = df['time'].apply(lambda x: x.get('cpu_time', None) if isinstance(x, dict) else None)
    df['average_time'] = df['time'].apply(lambda x: x.get('average_time', None) if isinstance(x, dict) else None)
    
    df.drop(columns=['time'], inplace=True)
    
    # Si 'memory' es None, esto se maneja aquí
    df['memory_used'] = df['memory'].apply(lambda x: x[0] if x is not None and isinstance(x, list) else 0)
    
    return df



def plot_tests_cache(df: pd.DataFrame):
    """Function to plot the results of the cache tests."""
    
    # Verificar que el DataFrame no esté vacío
    if df.empty:
        print("El DataFrame está vacío. No hay datos para graficar.")
        return

    # Filtrar por operaciones de tipo 'select'
    select_df = df[df['op_type'] == 'select']

    # Configurar la figura
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)  # Ajustar el número de subgráficas según el número de DBs
    db_names = select_df['db_name'].unique()  # Obtener los nombres únicos de las bases de datos

    for i, db in enumerate(db_names):
        db_df = select_df[select_df['db_name'] == db]

        # Verificar que haya datos para graficar
        if db_df.empty:
            print(f"No hay datos para graficar en la base de datos: {db}")
            continue  # Saltar si no hay datos

        # Traza líneas para pruebas sin caché
        sns.lineplot(data=db_df[db_df['cache_used'] == 'No'], 
                     x='data_size', 
                     y='cpu_time', 
                     ax=axes[i], 
                     label='CPU Time (Sin Caché)', 
                     color='blue', 
                     marker='o')  # Agregar marcadores para mayor claridad
        
        sns.lineplot(data=db_df[db_df['cache_used'] == 'No'], 
                     x='data_size', 
                     y='real_time', 
                     ax=axes[i], 
                     label='Real Time (Sin Caché)', 
                     linestyle='--', 
                     color='lightblue', 
                     marker='o')

        # Traza líneas para pruebas con caché
        sns.lineplot(data=db_df[db_df['cache_used'] == 'Yes'], 
                     x='data_size', 
                     y='cpu_time', 
                     ax=axes[i], 
                     label='CPU Time (Con Caché)', 
                     color='orange', 
                     marker='o')
        
        sns.lineplot(data=db_df[db_df['cache_used'] == 'Yes'], 
                     x='data_size', 
                     y='real_time', 
                     ax=axes[i], 
                     label='Real Time (Con Caché)', 
                     linestyle='--', 
                     color='yellow', 
                     marker='o')

        # Configuraciones de los ejes
        axes[i].set_title(f'Select in {db}')
        axes[i].set_xlabel('Data Size')
        axes[i].set_ylabel('Time per Data Size (seconds)')
        axes[i].legend(title="Cache Usage")  # Leyenda mejorada

    # Ajustar el límite del eje y
    max_time = select_df[['cpu_time', 'real_time']].max().max() * 1.1
    for ax in axes:
        ax.set_ylim(0, max_time)  # Ajustar límite de todos los ejes Y

    plt.suptitle('Select Operations - Time vs Data Size', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar el layout
    plt.show()

    return fig



def run_tests_update(db_names: List[str], params: List[Dict[str, str]], sizes: List[int],
                     provider: MasterProvider, op_type: str = 'update') -> pd.DataFrame:
    """
    Executes tests with different data sizes and measures time and memory for UPDATE operations.
    
    Args:
        db_names (list): Names of the databases (['postgres', 'sqlite', 'mongo', 'duck']).
        params (list): Connection parameters for each database.
        sizes (list): Data sizes to test.
        provider: Data provider that generates users and vehicles.
        queries (dict): List of queries to execute in each database.
        op_type (str): Type of operation to measure ('update').
        method (str): Method to process individual records ('one_each' or 'at_once').

    Returns:
        pd.DataFrame: DataFrame containing the results of the tests.
    """
    
    results = {
        'data_size': [],
        'db_name': [],
        'op_type': [],
        'method': [],
        'real_time': [],
        'cpu_time': [],
    }

    for size in sizes:

        print('\n\nStep 1: Generating users and vehicles...')
        us, veh = provider.generate(size, int(size * 2))  # Generate users and vehicles
        create_csv(['out/csv/users.csv', 'out/csv/vehicles.csv'], [us, veh])
        create_json(['out/json/users.json', 'out/json/vehicles.json'], [us, veh])

        print('\n\nStep 2: Inserting in the db')
        for name, param in zip(db_names, params):
            conn = get_connection(name, param)
            delete_data(name, param, ['usuarios', 'vehiculos'])
            insert_data(name, param, 'out/csv/users.csv', 'usuarios', method='at_once')
            insert_data(name, param, 'out/csv/vehicles.csv', 'vehiculos', method='at_once')
            conn.close()

        print(f'\n\nRunning tests with data size: {size}\n{"----" * 10}')
    
        for name, param in zip(db_names, params):
            print('---------' * 10)
            print(f'Running tests in: {name}')
            records = get_record_keys(name, param, 'usuarios')

            # Measure updates with open connection
            conn = get_connection(name, param)
            print('----- Running test keeping conexion alive -----')
            update_function_open = partial(update_records_open_connection, name, conn, param, records)
            time_taken_open = time_function(update_function_open, op_type=op_type)
            conn.close()

            # Store results for open connection
            results['data_size'].append(size)
            results['db_name'].append(name)
            results['op_type'].append(op_type)
            results['method'].append('open_connection')
            results['real_time'].append(time_taken_open['real_time'])
            results['cpu_time'].append(time_taken_open['cpu_time'])


            # Measure updates with close connection
            print('----- Running test without keeping conexion alive -----')
            time_taken_close = time_function(partial(update_records_close_connection, name, param, records), op_type=op_type)

            # Store results for close connection
            results['data_size'].append(size)
            results['db_name'].append(name)
            results['op_type'].append(op_type)
            results['method'].append('close_connection')
            results['real_time'].append(time_taken_close['real_time'])
            results['cpu_time'].append(time_taken_close['cpu_time'])

        print('---------' * 10)

    return pd.DataFrame(results)


def plot_test_update(df):
    """Function to plot the results of the update tests."""

    db_names = df['db_name'].unique()
    num_dbs = len(db_names)

    fig, axes = plt.subplots(nrows=1, ncols=num_dbs, figsize=(15, 5), sharey=True)
    
    if num_dbs == 1:
        axes = [axes]  

    for i, db in enumerate(db_names):
        ax = axes[i]
        db_data = df[df['db_name'] == db]
        
        sns.lineplot(
            x='data_size', 
            y='real_time', 
            hue='method', 
            style='method',
            markers=True,  # Añadir marcadores a los puntos de datos
            data=db_data, 
            ax=ax,
            palette='Set2'
        )
        ax.set_title(f'{db} - Real Time')
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Real Time (seconds)')
        ax.legend(title='Method')

    plt.tight_layout()
    plt.show()

    return fig

