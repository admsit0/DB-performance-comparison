############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/cache_implementation.py
# Due Date: 22/October/2024
# Description: lib implementing all the classes and functions relative to the cache system of the project
############################################################################################################

import time
from pymemcache.client import base
from pymemcache import serde # Pickle serializer

class SimpleCache:
    """Simple cache implementation using a dictionary."""
    
    def __init__(self):
        self.cache = {}

    def set(self, key, value, expire=None):
        """Sets a value in the cache with a key and an expiration time (expire => ttl)."""
        if expire is not None:
            expiration_time = time.time() + expire
            self.cache[key] = (value, expiration_time)
        else:
            self.cache[key] = (value, None)

    def get(self, key):
        """Retrieves a value from the cache. Returns None if the key does not exist or has expired."""
        if key in self.cache:
            value, expiration_time = self.cache[key]
            if expiration_time is None or time.time() < expiration_time:
                return value
            else:
                # If it has expired, remove it from the cache
                del self.cache[key]
                return None
        return None

    def clear(self):
        """Clears the entire cache."""
        self.cache.clear()


def execute_query_and_cache_relational(query, conn, key, cache, cache_time=60):
    """Executes a SQL query and caches the result.
    
    Args:
        query (str): The SQL query to execute.
        conn: The connection to the relational database.
        cache: Cache object.
        cache_time (int): Cache time in seconds.

    Returns:
        Query results.
    """
    
    result = cache.get(key)

    if not result:
        try:
            # If not in cache, execute query and save result
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()  # Retrieve all results of the query
            cache.set(key, result, expire=cache_time)
        except Exception as e:  # General exception to catch all DB errors
            print(f"Error executing query: {e}")
            result = None  # In case of error, return None or an appropriate value
        finally:
            if cursor: 
                cursor.close()  # Close the cursor after execution
    
    return result


def execute_query_and_cache_mongo(query, coll, key, cache, cache_time=60):
    """Executes a query in MongoDB and caches the result.

    Args:
        query (dict or list): The MongoDB query to execute. If it's a list, it assumes an aggregation pipeline.
        mongo_conn: The connection to the MongoDB database.
        cache: Cache object.
        cache_time (int): Cache time in seconds.

    Returns:
        Query results.
    """
    
    result = cache.get(key)
    if not result:
        # Si la consulta es una lista, asumimos que es una pipeline de agregación
        if isinstance(query, list):
            result = list(coll.aggregate(query))
        else:
            result = list(coll.find(query))  # Ejecutar como consulta simple si no es pipeline
    cache.set(key, result, expire=cache_time)

    return result
