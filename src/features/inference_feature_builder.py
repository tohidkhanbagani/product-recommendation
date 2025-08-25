from src.data_ingest.data_ingest import Ingest
from src.utils.functions import ModelSolutions
from typing import List,Union,Tuple
import numpy as np
import pandas as pd



class InputModelFeatures:

    def __init__(self):
        
        self.ingest = Ingest()
        self.model_solutions = ModelSolutions()
        self.recommendation_data = self.ingest.data_ingest(folders=['data','processed'],file_name='recommendation_pivot_table.csv')
        self.recommend_data_schema = self.ingest.schema(choice='dictionary',obj=self.recommendation_data)
        self.processed_data_metrics = self.ingest.data_ingest(folders=['data','processed','processed_data'],file_name='processed_metrics.csv')
        self.product_list = list(self.recommend_data_schema.keys())
        self.product_to_index = {product: i for i, product in enumerate(self.product_list)}



    def product_recommendation_features(self,
                                        id:Union[int,List[int]]=None,
                                        product:Union[str, List[str]]=None,
                                        counts:Union[str, List[int]]=None
                                        ):
        
            # 1) Normalize inputs to lists
            if isinstance(product, str):
                product = [product]
            if isinstance(counts, int):
                counts = [counts]
            if len(product) != len(counts):
                raise ValueError("`product` and `counts` must be the same length")
            
            # 2) Prepare zero vector
            user_arr = np.zeros((1, len(self.product_list)))

            # 3) Find indices of each product
            product_indices = [self.product_to_index[p] for p in product if p in self.product_to_index]

            # 4) Double-check lengths
            if len(product_indices) != len(counts):
                raise ValueError(
                    f"Found {len(product_indices)} matching products but {len(counts)} counts"
                )

            # 5) Assign the counts
            for idx, cnt in zip(product_indices, counts):
                user_arr[0, idx] = cnt

            return id, user_arr
    
    def bulk_entries_recommendation(self,
                                data: pd.DataFrame = None,
                                fake: bool = False,
                                entries: int = 50,
                                min_id: int = 10000,
                                max_id: int = 20000,
                                purchase_size: int = 10,
                                replace: bool = True):
        if fake:
            # 1. Generate unique customer IDs
            ids = np.random.choice(range(min_id, max_id), size=entries, replace=False)
            result = np.zeros(shape=(entries, len(self.product_list)), dtype=int)

            # 2. Generate synthetic purchase patterns per customer
            for i, cust_id in enumerate(ids):
                values, counts = np.unique(
                    np.random.choice(self.product_list, size=purchase_size, replace=replace),
                    return_counts=True
                )
                _, sample = self.product_recommendation_features(id=cust_id, product=values, counts=counts)
                result[i] = sample

            return ids.tolist(), result

        else:
            ids = data['id'].unique().tolist()
            result = np.zeros(shape=(len(ids), len(self.product_list)), dtype=int)

            for i, cust_id in enumerate(ids):
                customer = data.query('id == @cust_id')
                products, counts = np.unique(customer['product'].to_list(), return_counts=True)
                _, sample = self.product_recommendation_features(id=cust_id, product=products, counts=counts)
                result[i] = sample

            return ids, result

    
