############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/data_schema.py
# Due Date: 22/October/2024
# Description: contains functions neccesary to define the databases schema before inserting data
############################################################################################################

from typing import List

def create_tables(conn, db_type: str ='default', commit: bool =False, table_names: List[str] =['usuarios', 'vehiculos']) -> None:
    
    """Function to create the tables and define the schema of the database.
    Valid db's are 'psql', 'sqlite' and 'duckdb'. MongoDB needs no schema definition."""

    def __drop_tables(cur, db_type, table_names):
        """Private internal function to previously drop tables if they already exist."""
        if db_type == 'psql':  # Slightly different syntax for postgres
            cur.execute(f"DROP TABLE IF EXISTS {table_names[1]} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {table_names[0]} CASCADE;")
        else:
            cur.execute(f"DROP TABLE IF EXISTS {table_names[1]};")  # Correct order to avoid FK constraint violation
            cur.execute(f"DROP TABLE IF EXISTS {table_names[0]};")

    # Using a context manager for connection
    with conn:
        cur = conn.cursor()  # Create a cursor for the connection

        # Drop existing tables
        __drop_tables(cur, db_type, table_names)

        # Create table usuarios
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_names[0]} (
            nombre VARCHAR(100) NOT NULL,
            dni VARCHAR(20) PRIMARY KEY,
            email VARCHAR(100) NOT NULL UNIQUE,
            telefono_movil VARCHAR(30) UNIQUE,
            telefono_fijo VARCHAR(30) UNIQUE,
            direccion VARCHAR(255),
            ciudad VARCHAR(100),
            codigo_postal VARCHAR(100),
            provincia VARCHAR(100)
        );""")

        # Create table vehiculos
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_names[1]} (
            matricula VARCHAR(15) UNIQUE,
            numero_bastidor VARCHAR(20) PRIMARY KEY,
            anno INTEGER,
            marca VARCHAR(100),
            modelo VARCHAR(100),
            categoria VARCHAR(50),
            dni_usuario VARCHAR(20) REFERENCES usuarios(dni)
        );""")

        # Commit changes if specified
        if commit:
            conn.commit()
