"""
Convert JSON data to human-readable form.
Usage:
  prettyJSON.py inputFile [outputFile]
"""

import sys

out = ','.join([line[:-1] for line in sys.stdin.readlines()])

sys.stdout.write('['+out+']')

"""
for line in sys.stdin:

    sys.stdout.write(line[:-1]+',')

sys.stdout.write(']')

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
""" 
