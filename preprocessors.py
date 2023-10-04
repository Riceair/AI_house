from preprocessing.normalize import *
from preprocessing.symbolic import *
from config.data_setting import *
import pandas as pd

class Preprocessor:
    def __init__(self,
                 data_path = train_path,
                 symbolic_names = symbolic_names,
                 numerical_names = numerical_names,
                 coordinate_names = coordinate_names,
                 truth_name = truth_name,
                 symbolic_type = "onehot",
                 num_min = -1, num_max = 1,):
        '''
        symbolic_type: symbolic type can be 'onehot', 'order', 'prob'\n
        data_path: data csv path \n
        symbolic_names: symbolic data column names \n
        numerical_names: numerical data column names \n
        coordinate_names: twd97 column names \n
        truth_name: truth data column name \n
        num_min/max: numerical data max/min value \n
        '''
        self.__setSymoblicType(symbolic_type)
        df = pd.read_csv(train_path)
        df = pd.DataFrame(df)

        # ground truth preprocessing
        self.y_true = df[truth_name].to_numpy()
        self.normalizer_true = Normalizer()
        self.y_true = self.normalizer_true.normalize(self.y_true, num_min, num_max)

    def __setSymoblicType(self, symoblic_type):
        if symoblic_type == "onehot":
            self.create_table = create_onehot_table
        elif symoblic_type == "order":
            self.create_table = create_order_table
        elif symoblic_type == "prob":
            self.create_table = create_prob_table
        else:
            raise NameError("symbolic_type must be 'onehot', 'order', or 'prob'")

if __name__ == "__main__":
    preprocessor = Preprocessor()