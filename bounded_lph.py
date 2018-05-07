import numpy as np

class BoundedLPH:
    def __init__(self, dim_bounds):
        self.dim_bounds = dim_bounds
        for dim in dim_bounds:
            min_value = dim[0]
            max_value = dim[1]
            if(min_value > max_value):
                raise ValueError("Minimum value {min_value} cannot be greater than max value {max_value}".format(min_value=min_value, max_value=max_value))

    def get_binary_hash(self, point, level):
        if(self.dim_bounds.shape[0] != point.shape[0]):
            raise ValueError("Point is not the same number of dimensions ({point_dim}) as the boundary definition ({bound_dim})".format(point_dim=point.shape[0], bound_dim=self.dim_bounds.shape[0]))

        #Loop through each dimension and get the hash values for that dimension
        hash_array = np.zeros((len(self.dim_bounds), level), dtype=int)
        for idx in range(0, len(self.dim_bounds)):
            dim = self.dim_bounds[idx]
            dim_value = point[idx]
            min_value = dim[0]
            max_value = dim[1]

            if(dim_value > max_value):
                raise ValueError("Value {dim_value} is greater than maximum {max_value}".format(dim_value=dim_value, max_value=max_value))

            if(dim_value < min_value):
                raise ValueError("Value {dim_value} is less than minimum {min_value}".format(dim_value=dim_value, min_value=min_value))

            self._get_dim_hash(dim_value = dim_value, level = level, min_value = min_value, max_value = max_value, hash_array = hash_array[idx])

        #Hash values are stored one row per dimension, we need to take all the values in column 1, then all the values in column 2, etc. 
        return np.reshape(hash_array, -1, order='F')
        

    #Recursively get the hash for a single dimension
    def _get_dim_hash(self, dim_value, level, min_value, max_value, hash_array):
        mid_value = (min_value + max_value) / 2

        print("Value {dim_value}, min {min_value}, mid {mid_value}, max {max_value}".format(dim_value=dim_value, min_value=min_value, mid_value=mid_value, max_value=max_value))
        if(dim_value > mid_value):
            current_hash = 1 
            min_value = mid_value
        else:
            current_hash = 0
            max_value = mid_value

        print("Setting index {index} to {current_hash}: ".format(index=-level, current_hash = current_hash))
        hash_array[-level] = current_hash

        #print(hash_array)

        if(level > 1):
            self._get_dim_hash(dim_value = dim_value, level = level - 1, min_value = min_value, max_value = max_value, hash_array = hash_array) 
        
