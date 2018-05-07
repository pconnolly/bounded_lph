import numpy as np
from bounded_lph import  BoundedLPH

class GeoHash(BoundedLPH):
    def __init__(self):
        geo_bounds = np.array([[-90,90], [-180, 180]])
        super().__init__(geo_bounds)

