import numpy as np
from bounded_lph import  BoundedLPH

class ColorHash(BoundedLPH):

    def __init__(self):
        #Boundary of values must be in this range. Data must be in the same column order as columns here
        bounds = np.array([[0, 100], [-128, 127], [-128,127]])

        #Number of characters in character set must be a power of 2
        charset = 'abcdefgh'

        super().__init__(bounds, charset)

