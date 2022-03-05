"""
Convert JSON data to human-readable form.
Usage:
  prettyJSON.py inputFile [outputFile]
"""

import sys
import numpy as np
import json
from sklearn.preprocessing import normalize
from scipy import sparse

alpha = 0.01
sparse_thr = 1e-3
ntpc = 40

def main(sysstdin):

    all_thetas = []

    try:
        for line in sysstdin:
            thetas = np.random.dirichlet(alpha*np.ones((ntpc,)), 1)
            thetas[thetas<sparse_thr] = 0
            thetas = normalize(thetas,axis=1,norm='l1')
            thetas = sparse.csr_matrix(thetas, copy=True)
            all_thetas.append(dict(zip([int(el) for el in thetas.indices], thetas.data)))

        return all_thetas
        
    except:
        pass


if __name__ == "__main__":

    sys.stdout.write(json.dumps(main(sys.stdin)))
    