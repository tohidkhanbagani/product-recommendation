import numpy as np
import pandas as pd
from typing import Union, List
from src.data_ingest.data_ingest import Ingest
from src.features.inference_feature_builder import InputModelFeatures
from src.model.product_recommender import ProductRecommender

class ModelInference:

    def __init__(self):
        self.main_data = Ingest()
        self.model_inputs = InputModelFeatures()
        self.product_recommender = ProductRecommender()
        self.product_list_array = np.array(self.model_inputs.product_list)

    
    def product_recommendation(self,
                               id:Union[int,List[int]]=None,
                               product:Union[str, List[str]]=None,
                               counts:Union[str, List[int]]=None,
                               multiple:bool=False,
                               N:int=10,
                               recalculate:bool=False,
                               userid:int=0,
                               filter_liked:bool=True,
                               fake:bool=False,
                               data:pd.DataFrame=None,
                               entries:int=50
                               ):
        if multiple==True:
            id, user_array = self.model_inputs.bulk_entries_recommendation(data=data,fake=fake,entries=entries)
            recommender_predictions = self.product_recommender.predict(user_arr=user_array,
                                                                                                         N=N,
                                                                                                         recalculate=recalculate,
                                                                                                         userid=userid,
                                                                                                         filter_liked=filter_liked,
                                                                                                         multiple=True
                                                                                                         )
            recommender_predictions=[self.product_list_array[indices].tolist() for indices in recommender_predictions]
            prices = []
            for i, j in enumerate(recommender_predictions):
                price = [float(self.model_inputs.processed_data_metrics.query('product==@product').iloc[0]['purchase_amount']) for product in j]
                prices.append(price)
            return id, recommender_predictions, prices
        else:
            id, user_array = self.model_inputs.product_recommendation_features(id=id,counts=counts,product=product)
            recommender_predictions, recommendation_confidance_scores = self.product_recommender.predict(user_arr=user_array,
                                                                                                         N=N,
                                                                                                         recalculate=recalculate,
                                                                                                         userid=userid,
                                                                                                         filter_liked=filter_liked
                                                                                                         )
            recommender_predictions= self.product_list_array[recommender_predictions].tolist()
            prices = [float(self.model_inputs.processed_data_metrics.query('product==@j').iloc[0]['purchase_amount']) for i,j in enumerate(recommender_predictions)]
            return recommender_predictions, prices, recommendation_confidance_scores.tolist()
