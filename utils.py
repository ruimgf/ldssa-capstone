from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
class LowerCase(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Fit the transformer and store it.
        return self
        
    def transform(self, X):
        # Transform X.
        X = X.copy()

        X['seat'] = X.seat.str.lower()
        X['person_attributes'] = X.person_attributes.str.lower()
        X['person_attributes'] = X.person_attributes.str.replace('in_stopped_vehicle','other')
        X['person_attributes'] = X.person_attributes.str.replace('other_on_foot','on_foot')
        X['person_attributes'] = X.person_attributes.str.replace('unknown_in_vehicle','driving')
        X['person_attributes'] = X.person_attributes.str.replace('unknown_in_other_vehicle_type','driving')
        X['other_person_info'] = ~X.other_person_location.isnull()
        X['is_m'] = np.where(X.m_or_f=='f',1,0)
        #X['other_person'] = np.where(~X.other_person_location.isnull(),1,0)
        #X['number_of_other_factores'] = np.where(~X.other_factor_1.isnull(),1,0) + np.where(~X.other_factor_2.isnull(),1,0) + np.where(~X.other_factor_3.isnull(),1,0)
        return X.drop(columns='m_or_f')

class DropUnusefull(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop = ['seat',
                             'other_person_location',
                             'other_factor_1',
                             'other_factor_2',
                             'other_factor_3'
                            ]
    def fit(self, X, y=None):
        # Fit the transformer and store it.
        return self
        
    def transform(self, X):
        # Transform X.
        return X.drop(columns=self.cols_to_drop)
