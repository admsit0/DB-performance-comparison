############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/connection.py
# Due Date: 22/October/2024
# Description: connection params for the different databases. EDIT HERE FOR YOUR OWN DATABASES OR PC
############################################################################################################


PSQL_PARAMS = {'host': 'localhost', 'database': 'postp01db', 'user': 'postgres', 'password': 'db'}
SQLITE_PARAMS = {'db': 'sqlite3p01db.db'}
MONGO_PARAMS = {'uri': 'mongodb://localhost:27017/', 'db_name': 'mongop01db', 'collection': 'usuarios'}
MONGO_PARAMS2 = {'uri': 'mongodb://localhost:27017/', 'db_name': 'mongop01db', 'collection': 'vehiculos'}
MONGO_PARAMS3 = {'uri': 'mongodb://localhost:27017/', 'db_name': 'mongop01db', 'collection': 'mix'}
DUCK_PARAMS = {'db': 'duckdbp01db.db'}