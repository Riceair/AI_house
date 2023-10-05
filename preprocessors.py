from pyproj import Transformer as p_Transformer
from preprocessing.normalize import *
from preprocessing.symbolic import *
from config.data_setting import *
import pandas as pd

class Preprocessor:
    def __init__(self,
                 csv_path = train_path,
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
        coordinate_names: twd97 column names (0: horiz, 1: vert) \n
        truth_name: truth data column name \n
        num_min/max: numerical data max/min value \n
        '''
        self.symbolic_names = symbolic_names
        self.numerical_names = numerical_names
        self.coordinate_names = coordinate_names
        self.num_min = num_min
        self.num_max = num_max
        self.symbolic_type = symbolic_type
        self.__setSymoblicType(self.symbolic_type)
        df = pd.read_csv(csv_path)
        df = pd.DataFrame(df)

        # ground truth normalizer
        self.y_true = df[truth_name].to_numpy()
        self.normalizer_true = Normalizer(self.y_true, self.num_min, self.num_max)
        self.y_true = self.normalizer_true.normalize(self.y_true)

        # symbolic columns encoder
        self.encoder_symb_dict = dict() # 儲存 symbolic 資料型態的 encoder
        for name in self.symbolic_names:
            symb_col = df[name].fillna("None") # 讀column，並把nan改成'None'
            symb_col = symb_col.to_numpy()
            table = self.create_table(symb_col)
            encoder = SymbolicEncoder(table)
            self.encoder_symb_dict[name] = encoder # 紀錄 symbolic encoder
        
        # numerical columns normalizers
        self.normalizer_num_dict = dict()
        for name in self.numerical_names:
            num_col = df[name].to_numpy()
            normalizer = Normalizer(num_col, self.num_min, self.num_max)
            self.normalizer_num_dict[name] = normalizer # 紀錄 numerical normalizer

        # coordinate columns transformer (twd97 -> lon, lat)
        self.coord_transformer = p_Transformer.from_crs("EPSG:3826", "EPSG:4326")  # TWD97 to WGS84
        self.normalizer_coord_dict = dict()
        h_col = df[coordinate_names[0]].to_numpy()
        v_col = df[coordinate_names[1]].to_numpy()
        lons, lats = self.__convertTwd97toWgs84(h_col, v_col)
        self.normalizer_coord_dict[coordinate_names[0]] = Normalizer(lons, self.num_min, self.num_max)
        self.normalizer_coord_dict[coordinate_names[1]] = Normalizer(lats, self.num_min, self.num_max)

    def preprocess(self, csv_path):
        df = pd.read_csv(csv_path)
        df = pd.DataFrame(df)

        # encode symbolic columns
        x_symbs = []
        for name in self.symbolic_names:
            encoder = self.encoder_symb_dict[name]
            col = df[name].fillna("None") # 讀column，並把nan改成'None'
            col = col.to_numpy()
            col = np.array(encoder.encode(col)) # encode the data
            x_symbs.append(col)
        # 串接 columns
        if self.symbolic_type == 'onehot':
            x_symbs = np.hstack(x_symbs)
        else:
            x_symbs = np.stack(x_symbs, axis=1)
        
        # normalize numerical columns
        x_nums = []
        for name in self.numerical_names:
            normalizer = self.normalizer_num_dict[name]
            col = df[name].to_numpy()
            col = normalizer.normalize(col)
            col = np.where(col<self.num_min, self.num_min, col) # 控制最大、最小值
            col = np.where(col>self.num_max, self.num_max, col)
            x_nums.append(col)
        x_nums = np.stack(x_nums, axis=1)

        # normalize coordinate
        x_coords = []
        h_col = df[coordinate_names[0]].to_numpy()
        v_col = df[coordinate_names[1]].to_numpy()
        lons, lats = self.__convertTwd97toWgs84(h_col, v_col)
        normalizer_lon = self.normalizer_coord_dict[coordinate_names[0]] # get normalizer
        normalizer_lat = self.normalizer_coord_dict[coordinate_names[1]]
        lons = normalizer_lon.normalize(lons)
        lats = normalizer_lat.normalize(lats)
        lons = np.where(lons<self.num_min, self.num_min, lons) # 控制最大、最小值
        lons = np.where(lons>self.num_max, self.num_max, lons)
        lats = np.where(lats<self.num_min, self.num_min, lats)
        lats = np.where(lats>self.num_max, self.num_max, lats)
        x_coords = np.stack([lons, lats], axis=1)

        x = np.hstack([x_symbs, x_nums, x_coords])
        return x

    def __convertTwd97toWgs84(self, h_col, v_col):
        lons, lats = [], []
        for h, v in zip(h_col, v_col):
            lon, lat = self.coord_transformer.transform(h, v)
            lons.append(lon)
            lats.append(lat)
        return np.array(lons), np.array(lats)

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
    preprocessor = Preprocessor(train_path, symbolic_type='onehot')
    x = preprocessor.preprocess(public_path)
    print(x.shape)