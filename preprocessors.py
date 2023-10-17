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
                 external_paths = external_paths,
                 truth_name = truth_name,
                 symbolic_type = "onehot",
                 num_min = 0, num_max = 1,
                 y_min = 0, y_max = 1,):
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
        self.external_paths = external_paths
        self.num_min = num_min
        self.num_max = num_max
        self.y_min = y_min
        self.y_max = y_max
        self.symbolic_type = symbolic_type
        df = pd.read_csv(csv_path)
        df = pd.DataFrame(df)

        # ground truth normalizer
        self.y_true = df[truth_name].to_numpy()
        self.normalizer_y = Normalizer(self.y_true, self.y_min, self.y_max)
        self.y_true = self.normalizer_y.normalize(self.y_true)

        # symbolic columns encoder
        self.encoder_symb_dict = dict() # 儲存 symbolic 資料型態的 encoder
        for name in self.symbolic_names:
            symb_col = df[name].fillna("None") # 讀column，並把nan改成'None'
            symb_col = symb_col.to_numpy()
            if self.symbolic_type == "onehot":
                table = create_onehot_table(symb_col, self.num_min, self.num_max)
            elif self.symbolic_type == "order":
                table = create_order_table(symb_col)
            elif self.symbolic_type == "y_order":
                table = create_y_order_table(symb_col, self.y_true, self.num_min, self.num_max)
            elif self.symbolic_type == "prob":
                table = create_prob_table(symb_col)
            else:
                raise NameError("symbolic_type must be 'onehot', 'order', 'y_order', or 'prob'")
            encoder = SymbolicEncoder(table)
            self.encoder_symb_dict[name] = encoder # 紀錄 symbolic encoder
        
        # numerical columns normalizers
        self.normalizer_num_dict = dict()
        for name in self.numerical_names:
            num_col = df[name].to_numpy()
            normalizer = Normalizer(num_col, self.num_min, self.num_max)
            self.normalizer_num_dict[name] = normalizer # 紀錄 numerical normalizer

        # coordinate columns transformer (twd97 -> lat, lng)
        self.coord_transformer = p_Transformer.from_crs("EPSG:3826", "EPSG:4326")  # TWD97 to WGS84
        self.normalizer_coord_dict = dict()
        h_col = df[coordinate_names[0]].to_numpy()
        v_col = df[coordinate_names[1]].to_numpy()
        lats, lngs = self.__convertTwd97toWgs84(h_col, v_col)

        # read external
        ex_lats, ex_lngs = self.__getExLatLng(self.external_paths)
        self.ex_coords_list = self.__getExCoordsList(self.external_paths) # 取得所有種類的 external 座標

        # get min distance normalizer
        min_ex_dists = self.__getMinExDist(lats, lngs, self.ex_coords_list)
        self.normalizer_dist = Normalizer(min_ex_dists, self.num_min, self.num_max)

        # get coordinate normalizer
        lats = np.hstack([lats, ex_lats])
        lngs = np.hstack([lngs, ex_lngs])
        self.normalizer_coord_dict["lat"] = Normalizer(lats, self.num_min, self.num_max)
        self.normalizer_coord_dict["lng"] = Normalizer(lngs, self.num_min, self.num_max)

    def preprocess(self, csv_path, is_external=False):
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
        if self.symbolic_type == 'onehot' or self.symbolic_type == 'y_order':
            x_symbs = np.hstack(x_symbs)
        else:
            x_symbs = np.stack(x_symbs, axis=1)
        
        # normalize numerical columns
        x_nums = []
        for name in self.numerical_names:
            normalizer = self.normalizer_num_dict[name]
            col = df[name].to_numpy()
            col = normalizer.normalize(col)
            col = np.clip(col, self.num_min, self.num_max) # 控制最大、最小值
            x_nums.append(col)
        x_nums = np.stack(x_nums, axis=1)

        # normalize coordinate
        h_col = df[self.coordinate_names[0]].to_numpy()
        v_col = df[self.coordinate_names[1]].to_numpy()
        lats, lngs = self.__convertTwd97toWgs84(h_col, v_col)
        normalizer_lat = self.normalizer_coord_dict["lat"] # get normalizer
        normalizer_lng = self.normalizer_coord_dict["lng"]
        normed_lats = normalizer_lat.normalize(lats)
        normed_lngs = normalizer_lng.normalize(lngs)
        normed_lats = np.clip(normed_lats, self.num_min, self.num_max) # 控制最大、最小值
        normed_lngs = np.clip(normed_lngs, self.num_min, self.num_max)
        x_coords = np.stack([normed_lats, normed_lngs], axis=1)

        if is_external:
            ex_dists = self.__getMinExDist(lats, lngs, self.ex_coords_list)
            ex_dists = self.normalizer_dist.normalize(ex_dists)
            ex_dists = np.clip(ex_dists, self.num_min, self.num_max)
            x = np.hstack([x_symbs, x_nums, x_coords, ex_dists])
        else:
            x = np.hstack([x_symbs, x_nums, x_coords])

        return x

    def denormalizeY(self, y_pred):
        return self.normalizer_y.denormalize(y_pred)

    def __getMinExDist(self, lats, lngs, ex_coords_list):
        def getMinDist(coord, ex_coords): # 取得距離最短數據
            coord_tile = np.tile(coord, (len(ex_coords), 1)) # 複製 n 次
            # 歐式距離計算 (用numpy矩陣計算加快速度)
            coord_div = coord_tile - ex_coords
            coord_squ = np.square(coord_div)
            coord_add = coord_squ[:,0] + coord_squ[:,1]
            dists = np.sqrt(coord_add)
            return dists.min()

        min_ex_dists = []
        for lat, lng in zip(lats, lngs):
            coord = np.array((lat, lng))
            min_dists = [getMinDist(coord, ex_coords) for ex_coords in ex_coords_list]
            min_ex_dists.append(np.array(min_dists))
        min_ex_dists = np.array(min_ex_dists)
        return min_ex_dists

    def __convertTwd97toWgs84(self, h_col, v_col):
        lats, lngs = [], []
        for h, v in zip(h_col, v_col):
            lat, lng = self.coord_transformer.transform(h, v)
            lats.append(lat)
            lngs.append(lng)
        return np.array(lats), np.array(lngs)

    def __getExLatLng(self, external_paths):
        ex_lats, ex_lngs = [], []
        for path in external_paths:
            df = pd.read_csv(path)
            df = pd.DataFrame(df)
            lats = df["lat"].to_list()
            lngs = df["lng"].to_list()

            # 去除不合規定的座標
            for lat, lng, in zip(lats, lngs):
                if pd.isna(lat) or pd.isna(lng):
                    continue
                if lat < 0 or lng < 0:
                    continue
                ex_lats.append(lat)
                ex_lngs.append(lng)

        return np.array(ex_lats), np.array(ex_lngs)
    
    def __getExCoordsList(self, external_paths):
        ex_coords_list = []
        for path in external_paths:
            df = pd.read_csv(path)
            df = pd.DataFrame(df)
            lats = df["lat"].to_list()
            lngs = df["lng"].to_list()
            coords = []
            # 去除不合規定的座標
            for lat, lng, in zip(lats, lngs):
                if pd.isna(lat) or pd.isna(lng):
                    continue
                if lat < 0 or lng < 0:
                    continue
                coords.append([lat, lng])
            ex_coords_list.append(np.array(coords))

        return ex_coords_list
       
if __name__ == "__main__":
    preprocessor = Preprocessor(train_path, symbolic_type='y_order')
    x = preprocessor.preprocess(public_path, is_external=True)
    print(x.shape)

    y = preprocessor.y_true
    print(np.max(y), np.min(y))
    y = preprocessor.denormalizeY(y)
    print(np.max(y), np.min(y))