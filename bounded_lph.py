import numpy as np
import math

class BoundedLPH:
    #Tests if a number is a power of 2
    def is_power2(self, num):
        return num != 0 and ((num & (num - 1)) == 0)


    def __init__(self, dim_bounds, charset):
        self.dim_bounds = dim_bounds
        for dim in dim_bounds:
            min_value = dim[0]
            max_value = dim[1]
            if(min_value > max_value):
                raise ValueError("Minimum value {min_value} cannot be greater than max value {max_value}".format(min_value=min_value, max_value=max_value))

        num_chars = len(charset)
        if(not self.is_power2(num_chars)):
            raise ValueError("Number of encoded values ({num_chars}) must be a power of 2".format(num_chars=num_chars))

        self.encodemap = {}
        for i in range(num_chars):
            self.encodemap[i] = charset[i]

        self.num_encoded_bits = (num_chars - 1).bit_length()

        if(self.num_encoded_bits > 8):
            raise ValueError("Number of encoded values ({num_chars}) exceeds maximum 64".format(num_chars=num_chars))


        #1-dim array of value for 2^index
        self.bit_power = 2**np.arange(self.num_encoded_bits, dtype = np.uint64)[::-1] 
        #print("bit_power")
        #print(self.bit_power)

    # There are probably faster ways to calculate this, e.g. https://www.factual.com/blog/how-geohashes-work/
    def get_encoded_hash(self, point, level):
        if(self.dim_bounds.shape[0] != point.shape[0]):
            raise ValueError("Point {point} is not the same number of dimensions ({point_dim}) as the boundary definition ({bound_dim})".format(point=point, point_dim=point.shape[0], bound_dim=self.dim_bounds.shape[0]))

        #print("level")
        #print(level)
        #print("num bits")
        #print(self.num_encoded_bits)
        num_bits_per_dim = math.ceil((level * self.num_encoded_bits) / len(self.dim_bounds))

        #Loop through each dimension and get the hash values for that dimension
        hash_array = np.zeros((len(self.dim_bounds), num_bits_per_dim), dtype=int)
        for idx in range(0, len(self.dim_bounds)):
            dim = self.dim_bounds[idx]
            dim_value = point[idx]
            min_value = dim[0]
            max_value = dim[1]

            if(dim_value > max_value):
                raise ValueError("Value {dim_value} is greater than maximum {max_value}".format(dim_value=dim_value, max_value=max_value))

            if(dim_value < min_value):
                raise ValueError("Value {dim_value} is less than minimum {min_value}".format(dim_value=dim_value, min_value=min_value))

            self._get_dim_hash(dim_value = dim_value, num_bits_per_dim = num_bits_per_dim, min_value = min_value, max_value = max_value, hash_array = hash_array[idx])

        #print("Hash array before: ")
        #print(hash_array)
        #print("num dims:" )
        #print(len(self.dim_bounds))
        #print("level")
        #print(level)
        #print("num encoded bits")
        #print(self.num_encoded_bits)

        #Hash values are stored one row per dimension, we need to take all the values in column 1, then all the values in column 2, etc. 
        linear_bits = np.reshape(hash_array, -1, order='F')

        #print("linear bits")
        #print(linear_bits)

        encoded_hash = ""
        for i in range(0, level):
            char_bits = linear_bits[i * self.num_encoded_bits:(i + 1) * self.num_encoded_bits]
            scaled_bits = np.multiply(char_bits, self.bit_power)
            decoded_value = scaled_bits.sum()
            #print("Bits:")
            #print(char_bits)
            #print(scaled_bits)
            #print(decoded_value)
            #print(self.encodemap[decoded_value])
            encoded_hash = encoded_hash + self.encodemap[decoded_value]


        return encoded_hash 

    #def get_binary_hash(self, point, level):
        #if(self.dim_bounds.shape[0] != point.shape[0]):
            #raise ValueError("Point is not the same number of dimensions ({point_dim}) as the boundary definition ({bound_dim})".format(point_dim=point.shape[0], bound_dim=self.dim_bounds.shape[0]))
#
        ##Loop through each dimension and get the hash values for that dimension
        #hash_array = np.zeros((len(self.dim_bounds), level), dtype=int)
        #for idx in range(0, len(self.dim_bounds)):
            #dim = self.dim_bounds[idx]
            #dim_value = point[idx]
            #min_value = dim[0]
            #max_value = dim[1]
#
            #if(dim_value > max_value):
                #raise ValueError("Value {dim_value} is greater than maximum {max_value}".format(dim_value=dim_value, max_value=max_value))
#
            #if(dim_value < min_value):
                #raise ValueError("Value {dim_value} is less than minimum {min_value}".format(dim_value=dim_value, min_value=min_value))
#
            #self._get_dim_hash(dim_value = dim_value, level = level, min_value = min_value, max_value = max_value, hash_array = hash_array[idx])
#
        ##Hash values are stored one row per dimension, we need to take all the values in column 1, then all the values in column 2, etc. 
        #return np.reshape(hash_array, -1, order='F')
        

    #Recursively get the hash for a single dimension
    def _get_dim_hash(self, dim_value, num_bits_per_dim, min_value, max_value, hash_array):
        mid_value = (min_value + max_value) / 2

        #print("Value {dim_value}, min {min_value}, mid {mid_value}, max {max_value}".format(dim_value=dim_value, min_value=min_value, mid_value=mid_value, max_value=max_value))
        if(dim_value > mid_value):
            current_hash = 1 
            min_value = mid_value
        else:
            current_hash = 0
            max_value = mid_value

        #print("Setting index {index} to {current_hash}: ".format(index=-num_bits_per_dim, current_hash = current_hash))
        hash_array[-num_bits_per_dim] = current_hash

        #print(hash_array)

        if(num_bits_per_dim > 1):
            self._get_dim_hash(dim_value = dim_value, num_bits_per_dim = num_bits_per_dim - 1, min_value = min_value, max_value = max_value, hash_array = hash_array) 


    def get_hash_centroid(self, hashcode):
        #TODO

    def get_hash_bbox(self, hashcode):
        #TODO
