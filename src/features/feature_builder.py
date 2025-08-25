import pandas as pd
import numpy as np
from scipy import sparse
from typing import Tuple

from src.data_ingest.data_ingest import Ingest
from src.utils.functions import ModelSolutions





class RecommendationData:
    def __init__(self, data_file: pd.DataFrame):
        self.data_file = data_file
        self.model_solutions = ModelSolutions()

    def split_train_test(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.data_file[['id', 'product', 'purchase_amount']].copy()

        # Random seed for reproducibility
        np.random.seed(42)

        # Create empty lists
        train_rows = []
        test_rows = []

        # Split each user's purchase data
        for user_id, user_data in df.groupby('id'):
            user_data = user_data.sample(frac=1)  # Shuffle
            split_index = int(len(user_data) * train_ratio)

            train_rows.append(user_data.iloc[:split_index])
            test_rows.append(user_data.iloc[split_index:])

        train_df = pd.concat(train_rows)
        test_df = pd.concat(test_rows)

        return train_df, test_df
    
    def get_sparse_matrices(self,production:bool=False,save:bool=False) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
        train_df, test_df = self.split_train_test()
        if production:
            data_file = pd.concat([train_df,test_df],ignore_index=True)
            data_pivot = data_file.pivot_table(index='id', columns='product', values='purchase_amount', aggfunc='sum', fill_value=0)
            data_csr = sparse.csr_matrix(data_pivot.values)
            self.model_solutions.dump_data(folders=['data','processed','baked'],file_name='recommendation_data_pivot.csv',obj=data_pivot)
            return data_csr, data_pivot
        else:
            # Create user-item matrices
            train_pivot = train_df.pivot_table(index='id', columns='product', values='purchase_amount', aggfunc='sum', fill_value=0)
            test_pivot = test_df.pivot_table(index='id', columns='product', values='purchase_amount', aggfunc='sum', fill_value=0)

            # Convert to CSR
            train_csr = sparse.csr_matrix(train_pivot.values)
            test_csr = sparse.csr_matrix(test_pivot.values)
            return train_csr, test_csr, train_pivot, test_pivot


