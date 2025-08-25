from src.utils.functions import ModelSolutions
from scipy import sparse
import numpy as np


class ProductRecommender:

    def __init__(self,model_path:list[str]=['models','trained_models','product_recommender'],model_name:str='product_recommendation_model.pkl'):
        self.model_sol = ModelSolutions()
        self.model = self.model_sol.load_model(folders=model_path,file_name=model_name)
        self.pivot_df = self.model_sol.load_data(folders=['data','processed','baked'],file_name='recommendation_data_pivot.csv')
        self.csr_data = sparse.csr_matrix(self.pivot_df.values)
        self.num_users, self.num_items = self.pivot_df.shape
        self.ids = np.array(self.model_sol.load_data(folders=['data','processed','preprocessed_data'],file_name='preprocessed_data.csv')['id'])


    def conversion(self,user_arr:np.ndarray=None):
        rows = user_arr.shape[0]
        sample_dict = {}
        for i in range(rows):
            sample_dict[i] = sparse.csr_matrix(user_arr[i])
        stacked = sparse.vstack(list(sample_dict.values()),format='csr')
        return stacked

    def predict(self,multiple:bool=False,user_arr:np.ndarray | sparse.csr_matrix=None,N:int=10,recalculate:bool=False,userid:int=0,filter_liked:bool=True):
        if user_arr.shape[0]>1:
            user_matrx = self.conversion(user_arr=user_arr)
        elif user_arr.shape[0]==1 and isinstance(user_arr,np.ndarray):
            user_csr = sparse.csr_matrix(user_arr)
        elif isinstance(user_arr,sparse.csr_matrix):
            user_csr = user_arr
        
        if multiple:
            recommendations = self.model.recommend_all(user_items=user_matrx,N=N,filter_already_liked_items=filter_liked)
            return recommendations
        else:
            if userid not in self.ids:
                product_idx, score = self.model.recommend(
                userid=userid,
                user_items=user_csr,
                N=N,
                filter_already_liked_items=filter_liked,
                recalculate_user=True)
                return product_idx, score
            else:
                product_idx, score = self.model.recommend(
                userid=userid,
                user_items=user_csr,
                N=N,
                filter_already_liked_items=filter_liked,
                recalculate_user=True)
                return product_idx, score