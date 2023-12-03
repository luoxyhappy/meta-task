import numpy as np
from utils2.cfa import CFA

# Example dataset
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 0, 1])  # 0 for majority class, 1 for minority class

cfa = CFA(fd=2, tol=0.1)
# Get synthetic data
X_cfa, y_cfa = cfa.run_cfa(X, y, get_synt_labels=False)
# OR
# Get synthetic data and label (synthetic = {0;1})
X_cfa, y_cfa, cfa_label = cfa.run_cfa(X, y, get_synt_labels=True)