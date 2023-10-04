import numpy as np

class Normalizer:
    def __init__(self):
        self.original_min = None
        self.original_max = None
        self.target_max = None
        self.target_min = None
        self.is_norm = False
    
    def normalize(self, values, target_min=0, target_max=1):
        values = np.array(values)
        self.is_norm = True
        self.target_min = target_min
        self.target_max = target_max
        self.original_min = np.min(values)
        self.original_max = np.max(values)
        norm_values = (values - self.original_min) / (self.original_max - self.original_min) *\
                        (self.target_max - self.target_min) + self.target_min
        return norm_values
    
    def denormalize(self, values):
        if not self.is_norm:
            raise NotImplementedError("Please run 'normalize' function first")
        values = np.array(values)
        denorm_values = (values - self.target_min) / (self.target_max - self.target_min) * \
                            (self.original_max - self.original_min) + self.original_min
        return denorm_values
        
if __name__=="__main__":
    values = [-2.5, -1, 2, 5, 8, 9.9, 10]
    normalizer = Normalizer()
    norm_values = normalizer.normalize(values, -1, 1)
    denorm_values = normalizer.denormalize(norm_values)
    print(norm_values)
    print(denorm_values)