import numpy as np


laplacian = np.array([
    [ 0,-1, 0],                   
    [-1, 4,-1],
    [ 0,-1, 0]]
)

vertical = np.array([
    [-1, 2,-1],                   
    [-1, 2,-1],
    [-1, 2,-1]]
)

horizontal= np.array([
    [-1,-1,-1],                   
    [ 2, 2, 2],
    [-1,-1,-1]]
)

diagonal_right = np.array([
    [-1,-1, 2],                   
    [-1, 2,-1],
    [ 2,-1,-1]]
)

diagonal_left = np.array([
    [ 2,-1,-1],                   
    [-1, 2,-1],
    [-1,-1, 2]]
)