import pandas as pd
import numpy as np


p = 0.5
cols = ['Age', 'Height', 'Weight', 'VKORC1 A/G', 'VKORC1 A/A', 'VKORC1 Missing', '*1/*1', '*1/*3', '*2/*2',
        '*2/*3', '*3/*3', 'Unknown Cyp2C9', 'Asian', 'Black or African American', 'Unknown Race',
        'Enzyme Inducer', 'Amiodarone (Cordarone)']

coeffs = [-0.2614, 0.0087, 0.0128, -0.8677, -1.6974, -0.4854, -0.5211, -0.9357, -1.0616, -1.9206, -2.3312, -0.2188,
          -0.1092, -0.2760, -0.1032, 1.1816, -0.5503]

const = 5.6044

for i in coeffs:
    lb = i - i*p
    ub = i + i*p
    print(lb, ub)
    print(np.random.uniform(lb, ub))