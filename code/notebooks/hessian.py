from sympy import *

L = Function("L")
Ld = Function("Ld", real=True)

q_prev, q_curr, q_next = symbols("q_{k-1}, q_k, q_{k+1}")

print(Ld(q_curr, q_next).is_real)

# def global_hessian_pattern(entries=2):
#     n = symbols('n', integer=True, positive=True)
#     N = symbols('N', integer=True, positive=True)
#     # k = symbols('k', integer=True, nonnegative=True)
    
#     D = lambda idx: MatrixSymbol(f'\\mathcal{{D}}_{idx}', n, n)
#     C = lambda idx: MatrixSymbol(f'C_{idx}', n, n)
    
#     hessian = [[0 for _ in range(2*entries + 1)] for _ in range(2*entries + 1)]
#     for k in range(entries):
#         i = k + 1
#         hessian[k][k] = D(i)
#         hessian[k + 1][k] = C(i).T
#         hessian[k][k + 1] = C(i)

#     for k in range(1, entries + 1):
#         i = f"{{N - {k}}}"
#         hessian[-k][-k] = D(i)
#         hessian[-k + 1][-k] = C(i).T
#         hessian[-k][-k + 1] = C(i)
    
#     return Matrix(hessian)

# H = global_hessian_pattern().subs(0, ' ')
# print(latex(H))



