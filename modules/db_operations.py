############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/db_operations.py
# Due Date: 22/October/2024
# Description: contains general functions to delete, insert, select and update data from the database
############################################################################################################

from tqdm import tqdm
from pymongo import ASCENDING
from modules.util import get_connection
from modules.cache_implementation import execute_query_and_cache_mongo, execute_query_and_cache_relational
import json
import pandas as pd
from typing import List, Dict, Any, Union, Optional

def delete_data(db_type: str, connection_params: Dict[str, str], table_names: List[str]) -> None:
    """
    Deletes all rows from the specified tables or collections in a database.
    
    Args:
        db_type (str): The type of database ('psql', 'sqlite', 'mongo', 'duck').
        connection_params (dict): Connection parameters for the database.
        table_names (list): List of table names (or collections for MongoDB) to delete content from.

    Note: important to delete in reverse order to avoid foreign key constraints violations.
    """
    conn = get_connection(db_type, connection_params)

    if db_type == 'mongo':
        db = conn[connection_params['db_name']]
        for collection in tqdm(table_names, desc=f"Deleting collections from db {db_type}"):
            db[collection].delete_many({})  # Delete all documents in the collection
            # print(f"All documents deleted from collection '{collection}' successfully.")
    else:
        cursor = conn.cursor()
        
        for table in tqdm(reversed(table_names), desc=f"Deleting tables from db {db_type}"):
            try:
                query = f"DELETE FROM {table};"
                cursor.execute(query)
                conn.commit()
                # print(f"All rows deleted from table '{table}' successfully.")
            except Exception as e:
                print(f"Error deleting rows from table '{table}': {e}")
                conn.rollback()
        
        cursor.close()
    
    conn.close()


def insert_data(db_type: str, connection_params: Dict[str, str], file_path: str, table_name: str, method: str = 'chunk', batch_size: int = 1000, use_index: bool = False) -> None:
    """
    Inserts data from a CSV or JSON file into the specified database.
    
    Args:
        db_type (str): The type of database ('psql', 'sqlite', 'duckdb', 'mongo').
        connection_params (dict): Connection parameters for the database.
        file_path (str): Path to the CSV or JSON file to insert data from.
        table_name (str): Name of the table (or collection for MongoDB).
        method (str): Insertion method ('chunk', 'one', 'at_once').
        batch_size (int): Number of rows to insert at a time for chunked operations.
        use_index (bool): If True, creates an index on the dni column.
    """
    
    # Get connection and initialize for MongoDB if needed
    

    if db_type not in ['psql', 'sqlite', 'mongo', 'duck']:
        raise ValueError("Unsupported database type")
    
    # Load data from the file (either CSV or JSON)
    if file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        data = df.to_dict(orient='records')
    else:
        raise ValueError("Unsupported file format. Only CSV and JSON are allowed.")
    
    # MongoDB insertion
    if db_type == 'mongo':
        conn = get_connection(db_type, connection_params)
        db = conn[connection_params['db_name']]
        mongo_collection = db.get_collection(table_name)  # Use get_collection instead of subscripting

        if method == 'at_once':
            conn = get_connection(db_type, connection_params)
            db = conn[connection_params['db_name']]
            mongo_collection = db.get_collection(table_name)  # Use get_collection instead of subscripting
            # Insert as much as possible (100,000 docs at a time)
            for i in tqdm(range(0, len(data), 100000), desc="Inserting at once in MongoDB"):
                mongo_collection.insert_many(data[i:i + 100000])
            conn.close()
        elif method == 'chunk':
            conn = get_connection(db_type, connection_params)
            db = conn[connection_params['db_name']]
            mongo_collection = db.get_collection(table_name)  # Use get_collection instead of subscripting
            # Insert data in user-specified chunks (batch_size)
            for i in tqdm(range(0, len(data), batch_size), desc=f"Inserting in chunks of {batch_size} in MongoDB"):
                mongo_collection.insert_many(data[i:i + batch_size])
            conn.close()
        elif method == 'one':
            # Insert one document at a time
            for doc in tqdm(data, desc="Inserting one by one in MongoDB"):
                conn = get_connection(db_type, connection_params)
                db = conn[connection_params['db_name']]
                mongo_collection = db.get_collection(table_name)  # Use get_collection instead of subscripting
                mongo_collection.insert_one(doc)
                conn.close()

        # Create index if requested
        if use_index:
            conn = get_connection(db_type, connection_params)
            db = conn[connection_params['db_name']]
            mongo_collection = db.get_collection(table_name)  # Use get_collection instead of subscripting
            mongo_collection.create_index([("dni", ASCENDING)])
            conn.close()
    
    # Relational DB insertion (Postgres, SQLite, DuckDB)
    else:
        
        if method == 'chunk':
            conn = get_connection(db_type, connection_params)
            cursor = conn.cursor()
            for i in tqdm(range(0, len(data), batch_size), desc=f"Inserting rows in chunks of {batch_size} in db {db_type}"):
                chunk = data[i:i + batch_size]
                placeholders = ', '.join(['%s'] * len(chunk[0])) if db_type == 'psql' else ', '.join(['?'] * len(chunk[0]))
                query = f"INSERT INTO {table_name} ({', '.join(chunk[0].keys())}) VALUES ({placeholders})"
                cursor.executemany(query, [tuple(row.values()) for row in chunk])
            conn.commit()
            cursor.close()
            conn.close()
        
        elif method == 'one':
            for row in tqdm(data, desc=f"Inserting rows one by one in db {db_type}"):
                conn = get_connection(db_type, connection_params)
                cursor = conn.cursor()
                placeholders = ', '.join(['%s'] * len(row)) if db_type == 'psql' else ', '.join(['?'] * len(row))
                query = f"INSERT INTO {table_name} ({', '.join(row.keys())}) VALUES ({placeholders})"
                cursor.execute(query, tuple(row.values()))
                conn.commit()
                cursor.close()
                conn.close()

        elif method == 'at_once':
            print(f"Inserting rows at_once in db {db_type}")
            conn = get_connection(db_type, connection_params)
            cursor = conn.cursor()
            placeholders = ', '.join(['%s'] * len(data[0])) if db_type == 'psql' else ', '.join(['?'] * len(data[0]))
            query = f"INSERT INTO {table_name} ({', '.join(data[0].keys())}) VALUES ({placeholders})"
            cursor.executemany(query, [tuple(row.values()) for row in data])
            conn.commit()
            #cursor.close()
            conn.close()

        if use_index:
            conn = get_connection(db_type, connection_params)
            cursor = conn.cursor()
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_dni ON {table_name} (dni);")
            cursor.close()
            conn.close()
        

def run_custom_select(db_type: str, query: Union[str, Dict[str, Any]], table: Optional[str] = None, params: Dict[str, str] = None) -> Optional[List[Any]]:
    """Generic function to execute a SELECT on various databases.
    
    Args:
        db_type (str): Type of database ('postgres', 'sqlite', 'mongo', 'duckdb').
        query (str or dict): SQL query string or MongoDB search criteria.
        table (str, optional): Table or collection name (required for MongoDB).
        params (dict): Connection parameters for the database.
    
    Returns:
        list: Query result as a list of rows or documents, or None if an error occurs.
    """
    conn = get_connection(db_type, params)
    result = None

    try:
        if db_type == 'mongo':
            db = conn[params['db_name']]
            collection = db[table]
            result = list(collection.find(query, {"_id": 0}))  # MongoDB find query, return documents
        else:
            cursor = conn.cursor()
            cursor.execute(query)  # SQL query for relational databases
            result = cursor.fetchall()
            cursor.close()

    except Exception as e:
        print(f"Error executing query on {db_type}: {e}")

    finally:
        conn.close()

    return result


def select_all_records(db_type: str, params: Dict[str, str], table: str) -> Optional[List[Any]]:
    """Select all records from a given table or collection in the specified database.
    
    Args:
        db_type (str): Type of database ('postgres', 'sqlite', 'mongo', 'duckdb').
        params (dict): Connection parameters for the database.
        table_or_collection (str): Name of the table or collection to query.
    
    Returns:
        list: All records from the table or collection, or None if an error occurs.
    """
    query = f"SELECT * FROM {table}" if db_type != 'mongo' else {}

    return run_custom_select(db_type, query, table, params)


def update_record(db_type: str, params: Dict[str, str], table_or_collection: str, field: str, new_value: Any, conditions: Union[str, Dict[str, Any]]) -> None:
    """Updates a record in a specified database based on conditions.

    Args:
        db_type (str): Type of database ('psql', 'sqlite', 'mongo', 'duck').
        params (dict): Connection parameters for the database.
        table_or_collection (str): Name of the table or collection.
        field (str): Field to update.
        new_value (Any): New value for the field.
        conditions (Union[str, Dict[str, Any]]): Conditions for the update.

    Raises:
        ValueError: If the database type is unsupported.
    """
    if db_type not in ['psql', 'sqlite', 'mongo', 'duck']:
        raise ValueError("Unsupported database type")

    conn = get_connection(db_type, params)

    try:
        if db_type in ['psql', 'sqlite', 'duck']:
            cursor = conn.cursor()
            placeholder = '%s' if db_type == 'psql' else '?'  # Use %s for PostgreSQL and ? for SQLite and DuckDB
            # Adjusting the condition string to use single quotes for string literals
            query = f"UPDATE {table_or_collection} SET {field} = {placeholder} WHERE {conditions.replace('\"', '\'')}"
            cursor.execute(query, (new_value,))  # Pass the new value as a parameter
            conn.commit()  # Commit changes
            cursor.close()  # Close cursor

        elif db_type == 'mongo':
            # For MongoDB, conditions should be a dict, not a string
            collection = conn[params['db_name']][table_or_collection]
            collection.update_many(conditions, {"$set": {field: new_value}})

    finally:
        conn.close()



def process_individual_records(db_type, param, record_keys, base_query, cache, method='one_each'):
    """
    Processes the results of an initial query one by one using two possible methods:
    'one_each' - Opens and closes the connection for each record.
    'at_once' - Opens a single connection and processes all records.

    Args:
        db_type (str): Type of database ('postgres', 'sqlite', 'mongo', etc.).
        param (dict): Connection parameters for each database.
        view_or_collection_name (str): Name of the view (relational) or collection (Mongo) from which to fetch records.
        cache (dict): Cache to store query results and avoid redundant queries.
        method (str): Execution method ('one_each' or 'at_once').
    """

    def query_by_dni(db_type: str, param: dict, dni: str, base_query: list, conn, cache):
        """Query a database by DNI and cache the results."""
        
        if db_type == 'mongo':
            db = conn[param['db_name']]
            coll = db[param['collection']]
            
            query_dni = {"$match": {"dni": dni}}
            
            # Asegúrate de que base_query sea una lista válida de etapas
            if not isinstance(base_query, list):
                base_query = [base_query]  # Convertir a lista si no lo es
                
            mongo_query = [query_dni] # Combina el filtro con el pipeline

            return execute_query_and_cache_mongo(mongo_query, coll, dni, cache)



    # Aquí se realizarían las consultas de acuerdo al método seleccionado
    if method == 'one_each':
        print(f"Processing records using method: {method}") 
        print(f'Running query: {base_query}')     
        for record_key in tqdm(record_keys):
            conn = get_connection(db_type, param)
            query_by_dni(db_type, param, record_key, base_query, conn, cache)
            conn.close()

    elif method == 'at_once':
        print(f"Processing records using method: {method}")
        print(f'Running query: {base_query}') 
        # Similar al anterior, pero procesas todos los registros en una sola conexión.
        if db_type == 'mongo':
            mongo_conn = get_connection(db_type, param)[param['db_name']]
            for record_key in tqdm(record_keys):
                query_by_dni(db_type, param, record_key, base_query, mongo_conn, cache)
            mongo_conn.client.close()
        else:
            conn = get_connection(db_type, param)
            for record_key in tqdm(record_keys):
                query_by_dni(db_type, param, record_key, base_query, conn, cache)
            conn.close()

    else:
        raise ValueError(f"Unknown method: {method}")



def get_record_keys(db_type, param, view_or_collection_name):
    """
    Retrieves record keys (DNIs) from the specified view or collection.

    Args:
        db_type (str): The type of database (e.g., 'mongo', 'postgresql', etc.)
        param (dict): Parameters for the database connection, including db_name.
        view_or_collection_name (str): The name of the view (for SQL) or collection (for MongoDB).

    Returns:
        list: A list of DNIs from the specified view or collection.
    """
    record_keys = []

    if db_type == 'mongo':
        # Conexión a MongoDB
        mongo_conn = get_connection(db_type, param)[param['db_name']]
        collection = mongo_conn[view_or_collection_name]

        # Obtener DNIs de la colección
        record_keys = [doc['dni'] for doc in collection.find({}, {'dni': 1})]  # Solo obtener el campo 'dni'
        mongo_conn.client.close()
    else:
        # Conexión a base de datos relacional
        conn = get_connection(db_type, param)
        cursor = conn.cursor()

        # Consultar DNIs de la vista
        query = f"SELECT dni FROM {view_or_collection_name}"  # Asumimos que hay una columna 'dni'
        try:
            cursor.execute(query)
            record_keys = [row[0] for row in cursor.fetchall()]  # Obtener todas las filas de resultados
        except Exception as e:
            print(f"Error retrieving record keys: {e}")
        finally:
            cursor.close()  # Cerrar el cursor
            conn.close()  # Cerrar la conexión

    return record_keys


def update_records_open_connection(db_name, conn, params, records):
    """Update records using an open connection."""
    for record in tqdm(records):
        if db_name == 'mongo':
            conn[params['db_name']][params['collection']].update_one(
                {'dni': record}, 
                {'$set': {'ciudad': 'Madriz'}}
            )
        else:  # For PostgreSQL, SQLite, and DuckDB
            update_query = f"UPDATE usuarios SET ciudad = 'Madriz' WHERE dni = '{record}';"
            conn.cursor().execute(update_query)


def update_records_close_connection(db_name, params, records):
    """Update records by opening and closing the connection for each update."""
    
    for record in tqdm(records):
        conn = get_connection(db_name, params)  # Open a new connection for each record
        if db_name == 'mongo':
            conn[params['db_name']][params['collection']].update_one(
                {'dni': record}, 
                {'$set': {'ciudad': 'Madriz'}}
                )
        else:
            update_query = f"UPDATE usuarios SET ciudad = 'Madriz' WHERE dni = '{record}';"
            conn.cursor().execute(update_query)
            conn.commit()  
            conn.close() 