import implicit
from xgboost import XGBClassifier
from typing import List,Any
from typing import Optional
import warnings
from sklearn.metrics import accuracy_score
from src.utils.functions import ModelSolutions
from src.data_ingest.data_ingest import Ingest
from src.features.feature_builder import RecommendationData

warnings.filterwarnings('ignore')



class TrainPipeline:

    def __init__(self,
                 data_folder:list[str]=['data','raw'],
                 data_file:str='E-Commerce_data.csv',
                 recommender_factors:int=50,
                 recommender_iterations:int=30
                 ):
        self.ingest = Ingest()
        self.model_solutions = ModelSolutions()
        self.raw_data = self.ingest.data_ingest(folders=data_folder,file_name=data_file)
        self.recomendation_data_init = RecommendationData(data_file=self.raw_data)
        self.recommender_model = implicit.als.AlternatingLeastSquares(factors=recommender_factors,iterations=recommender_iterations)

    def training_Product_recommender(self,save_path:List[str]=['models','trained_models','product_recommender'],model_name:str='product_recommendation_model.pkl'):
        data_csr, data_pivot = self.recomendation_data_init.get_sparse_matrices(production=True)
        self.recommender_model.fit(data_csr)
        self.model_solutions.save_model(folders=save_path,file_name=model_name,obj=self.recommender_model)
        return print(f"The Model Has been Trained and Saved at {'/'.join(save_path)}/{model_name}")
