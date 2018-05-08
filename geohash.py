import numpy as np
from bounded_lph import  BoundedLPH

class GeoHash(BoundedLPH):

    def __init__(self):
        geo_bounds = np.array([[-180, 180], [-90,90]])
        charset = '0123456789bcdefghjkmnpqrstuvwxyz'
        super().__init__(geo_bounds, charset)


    
