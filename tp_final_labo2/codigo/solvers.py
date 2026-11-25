import numpy as np

def jacobi(A, b, x0, tol, max_iter):
    """
    Jacobi: x_i^{(k+1)} = (b_i - sum_{j!=i} a_ij x_j^{(k)}) / a_ii
    A: csr_matrix, b: ndarray, x0: ndarray
    """
    n = len(b)
    x = x0.copy()
    x_new = np.zeros(n)
    diag = A.diagonal()
    R = A.copy()
    R.setdiag(0)
    R.eliminate_zeros()

    for k in range(max_iter):
        x_new = (b - R.dot(x)) / diag
        err = np.max(np.abs(x_new - x))
        x = x_new.copy()
        if err < tol:
            return x, k + 1, err
    return x, max_iter, err

def gauss_seidel(A, b, x0, tol, max_iter):
    """
    Gauss-Seidel: usa valores recién actualizados.
    Implementación con barrido por filas (lenta pero didáctica).
    """
    n = len(b)
    x = x0.copy()
    A = A.tocsr()
    indptr = A.indptr
    indices = A.indices
    data = A.data
    diag = A.diagonal()

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = 0.0
            for p in range(indptr[i], indptr[i+1]):
                j = indices[p]
                if j != i:
                    s += data[p] * x[j]
            x[i] = (b[i] - s) / diag[i]
        err = np.max(np.abs(x - x_old))
        if err < tol:
            return x, k + 1, err
    return x, max_iter, err

def jacobi_vectorizado(A, b, x0, tol, max_iter):
    """
    Jacobi vectorizado con NumPy/SciPy. Mucho mas rapido que GS con bucles.
    """
    n = len(b)
    x = x0.copy()
    diag = A.diagonal()
    R = A.copy()
    R.setdiag(0)
    R.eliminate_zeros()

    for k in range(max_iter):
        x_new = (b - R.dot(x)) / diag
        err = np.max(np.abs(x_new - x))
        x = x_new
        if err < tol:
            return x, k + 1, err
    return x, max_iter, err
