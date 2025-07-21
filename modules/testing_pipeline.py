############################################################################################################
# Authors: Adam Maltoni, Ibón de Mingo
# Project: BBDD - P1 - 3ero CEIDD
# Script: modules/testing_pipeline.py
# Due Date: 22/October/2024
# Description: contains full testing pipeline for the project, including plot saving and result exporting
############################################################################################################

from typing import List, Union, Dict

from matplotlib import pyplot as plt
import pandas as pd
from modules.data_generation import MasterProvider
from modules.testing import plot_test_data_generation, plot_test_db_operations, plot_test_update, run_tests_update, test_data_generation, test_db_operations
from modules.util import JOIN_QUERIES, PK_QUERIES, SELECT_ALL_QUERIES
from modules.connection import *
from modules.testing import run_tests_cache, plot_tests_cache
from modules.cache_implementation import SimpleCache, base, serde
import os

class AutoTester:
    """
    Class that contains the full testing pipeline with all the necessary different options"""

    def __init__(self, sizes: Union[Dict[str, List[int]] | List[List[int]]], initial_data_path = 'data/', 
                 plot_export_path = 'plots/', results_save_path = 'results/', generated_data_path = 'out/', 
                databases: list = ['psql', 'sqlite', 'mongo', 'duck'], parameters: list = [PSQL_PARAMS, SQLITE_PARAMS, MONGO_PARAMS, DUCK_PARAMS],
                 extra_params : List = [MONGO_PARAMS2, MONGO_PARAMS3], provider: Union[MasterProvider | None] = None,
                 iters: int = 1, custom_cache: SimpleCache = None, memcached: base.Client = None) -> None:
        
        self.initial_data_path = initial_data_path
        self.plot_export_path = plot_export_path
        self.generated_data_path = generated_data_path
        self.databases = databases
        self.plots = []
        self.results = []
        self.provider = MasterProvider(post_cod_path=initial_data_path+'cod_post_mun.csv', cars_path=initial_data_path+'coches.csv') if provider is None else provider
        self.results_save_path = results_save_path
        self.parameters = parameters
        self.extra_params = extra_params
        self.iters = iters
        self.custom_cache = custom_cache if custom_cache is not None else SimpleCache()
        self.mem_cache = memcached if memcached is not None else base.Client(('localhost', 11211), serde=serde.pickle_serde)

        if isinstance(sizes, dict):
            self.sizes = sizes
        elif isinstance(sizes, list) and len(sizes) == 4:
            self.sizes = {'huge': sizes[-1], 'big': sizes[-2], 'small': sizes[-3], 'ultra_small': sizes[-4]}
        else: raise ValueError("Sizes must be a dictionary with 4 kv pairs or a List of 4 List[int]")

    def __str__(self) -> str:
        return f"AutoTester with databases: {self.databases} and sizes: {self.sizes}"

    def adjust_settings(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __test_data_generation(self) -> tuple[pd.DataFrame, plt.Figure]:
        """Method to test the data generation functions"""

        res = test_data_generation(['Merging data', 'Creating CSVs', 'Creating JSONs'], self.sizes['huge'], self.provider,
                                    iters=self.iters, out_path=self.generated_data_path)
        
        fig = plot_test_data_generation(res)
        self.results.append(res)
        self.plots.append(fig)
        return res, fig

    def __test_operations_on_db(self) -> tuple[pd.DataFrame, plt.Figure]:
        """Method to test CRUD operations on the databases"""

        res = test_db_operations(self.sizes['big'], self.provider, iters=self.iters, databases=self.databases)
        fig = plot_test_db_operations(res)
        self.results.append(res)
        self.plots.append(fig)
        return res, fig
    

    def __tests_ops_cache(self, index: bool = False) -> tuple[List[pd.DataFrame], List[plt.Figure]]:
        """Function to run tests on cache operations with custom and Memcached caches, represents 12 tests in total"""

        def __run(db_names, params, sizes, provider, custom_cache, mem_cache, queries, methods, index: bool = False):
            """Private function to run tests with cache on various queries and methods"""

            dfs = []
            for method in methods:
                print(f"\nRunning tests with custom cache on {queries} queries selecting {method}\n")
                dfa = run_tests_cache(db_names=db_names, params=params, sizes=sizes, provider=provider, cache=custom_cache, queries=queries, method=method, index=index)

                print(f"\n\nRunning tests with Memcached on {queries} queries selecting {method}\n")
                dfb = run_tests_cache(db_names=db_names, params=params, sizes=sizes, provider=provider, cache=mem_cache, queries=queries, method=method, index=index)
                dfs.append(dfa)
                dfs.append(dfb)

            return dfs
        
        ms, res = ['one_each', 'at_once'], []
        special_params = self.parameters[:2] + [self.extra_params[1]] + self.parameters[3:] # PK queries need connection to 'mix' mongo collection
        
        # Join queries
        joins = __run(self.databases, self.parameters, self.sizes['small'], self.provider, self.custom_cache, self.mem_cache, JOIN_QUERIES, ms, index=index)

        # PK queries
        pks = __run(self.databases, special_params, self.sizes['small'], self.provider, self.custom_cache, self.mem_cache, PK_QUERIES, ms, index=index)

        # SELECT_ALL queries
        selects = __run(self.databases, self.parameters, self.sizes['small'], self.provider, self.custom_cache, self.mem_cache, SELECT_ALL_QUERIES, ms, index=index)

        for df in joins + pks + selects:
            res.append(df)
        figs = [plot_tests_cache(df) for df in res]
        self.results.extend(res)
        self.plots.extend(figs)
        return res, figs

    def __test_updates(self):
        """Function to run tests on update operations on databases"""
        
        df = run_tests_update(self.databases, self.parameters, self.sizes['ultra_small'], self.provider)    
        fig = plot_test_update(df)
        self.results.append(df)
        self.plots.append(fig)
        return df, fig


    def save_plots(self) -> None:
        """Export all the plots generated into the plot_export_path"""

        if not os.path.exists(self.plot_export_path):
            os.makedirs(self.plot_export_path)

        for idx, plot in enumerate(self.plots):
            ax = plot.gca()
            title = ax.get_title() if ax.get_title() else f'plot{idx}'
            file_path = os.path.join(self.plot_export_path, f'{title}.png')
            plot.savefig(file_path)


    def export_results(self) -> None:
        """Export into a text file all the results of the tests"""
        # Asegurarse de que el directorio de resultados exista
        os.makedirs(self.results_save_path, exist_ok=True)
        
        for i, res in enumerate(self.results):
            res.to_csv(self.results_save_path + f'result_{i}.csv')

    def run_tests(self):
        print('\n\nTESTING: Data generation with Faker (N users, 2N cars)\n')
        self.__test_data_generation()
        print('\n\nTESTING: CRUD operations on databases\n')
        self.__test_operations_on_db()
        print('\n\nTESTING: Cache operations, with custom and Memcached, on various queries, no indexes\n')
        self.__tests_ops_cache()
        print('\n\nTESTING: Re-running all queries relative to caché operations, but now with indexes\n')
        self.__tests_ops_cache(index=True)
        print('\n\nTESTING: Update operations on databases\n')
        self.__test_updates()


        self.save_plots()
        self.export_results()
        print('All tests have been run successfully, plots and results have been saved')
