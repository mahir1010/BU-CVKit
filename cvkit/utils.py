#: Magic number used to represent missing data
MAGIC_NUMBER = -4668
from collections.abc import Iterable
import numpy as np


def build_intrinsic(f,center,skew=[0,0]):
    f = (f,f) if not isinstance(f,Iterable) or len(f)<2 else f
    mat = np.zeros((3,3))
    mat[0,:]=[f[0],skew[0],center[0]]
    mat[1,:]=[skew[1],f[1],center[1]]
    mat[2,2]=1
    return mat
