import numpy as np
import matplotlib
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse.linalg import eigsh
import time
import pickle
#helper functions
def _fermi(beta, eps):
    if beta == 'inf':
        # T = 0 limit, μ = 0
        return (eps < 0).astype(float)
    # Stable Fermi function
    x = beta * eps
    # clip to avoid overflow in exp
    x = np.clip(x, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(x))
####################################
def _DOS_Integral(D,Z, beta,Nk=4000):
    """
    Returns 2*int de D(e) e n_F(Ze)
    Parameters
    ----------
    Z : float
        Bond renormalization factor multiplying ε in the Fermi function.
    D : float
        The half-width of the flat DoS  
    beta : float
        Inverse temperature.
    Nk : int
        Number of k points for the integral.
    """
    e = np.linspace(-D, +D, Nk, endpoint=True)
    integrand = e*_fermi(beta, Z * e)
    integral =  (1./(2*D))*np.trapezoid(integrand,e)
    return 2*integral

def _K_calculator(D,Z,beta):
    return np.sqrt(Z)*_DOS_Integral(D,Z,beta)
def _shift_calculator(D,Z,beta):
    return -0.5*Z*_DOS_Integral(D,Z,beta)

####################################
def _calc_thermal_obs(H, beta, O=None, return_Z=False,return_GS = False):
    """
    Compute <O>_beta = Tr[ O e^{-beta H} ] / Tr[ e^{-beta H} ]
    using dense diagonalization, with ground-state energy shift for stability.

    Parameters
    ----------
    H : scipy.sparse matrix or np.ndarray
        Hamiltonian (Hermitian).
    beta : float
        Inverse temperature.
    O : scipy.sparse matrix or np.ndarray or None
        Observable operator. If None, returns free energy pieces (or Z if return_Z).
    return_Z : bool
        If True, also return (Zpart, E0) where Zpart = Tr exp(-beta(H-E0)).#
    return_GS:  bool
        Skips all the above and just returns GS vector

    Returns
    -------
    obs : float
        Thermal expectation value <O>.
    (optional) Zpart : float
        Partition function with shifted energies, Zpart = sum_i exp(-beta (E_i - E0)).
    (optional) E0 : float
        Ground state energy (minimum eigenvalue).
    """
    # 1) Make dense arrays
    H_dense = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
    if O is not None:
        O_dense = O.toarray() if hasattr(O, "toarray") else np.asarray(O)
    else:
        O_dense = None

    # 2) Diagonalize (Hermitian)
    evals, evecs = np.linalg.eigh(H_dense)
    E0 = float(evals[0])

    #################################################
    if return_GS is True:
        #skip rest and return GS
        print(f'GS gap:{evals[1]-evals[0]}')
        return evals[0],evecs[:, 0]

    #################################################

    # 3) Shift energies by E0 to avoid under/overflow
    dE = evals - E0  # >= 0

    ############################################
    # if beta = 'inf' do GS calculation
    # NOTE that this doesn't treat degeneracies... ie always takes one state
    if beta == 'inf':
        if O_dense is None:
            if return_Z:
                return None, 1.0, E0
            return None

        evec_GS = evecs[:, 0]
        obs0 = np.einsum('i,ij,j->', np.conjugate(evec_GS), O_dense, evec_GS).real

        if return_Z:
            return obs0, 1.0, E0
        return obs0
    else:
        # 4) Log-safe weights: logw = -beta*dE
        # Use log-sum-exp for denominator
        logw = -beta * dE
        m = np.max(logw)  # should be 0, but keep robust
        w = np.exp(logw - m)  # scaled weights in [0,1]
        Z_scaled = np.sum(w)
        # True shifted partition function is Zpart = exp(m) * Z_scaled,
        # but exp(m)=1 typically. Keep the general formula:
        Zpart = np.exp(m) * Z_scaled

        if O_dense is None:
            if return_Z:
                return None, Zpart, E0
            return None

        # 5) Compute diagonal matrix elements <n|O|n> efficiently
        # O_nn = (V^† O V)_nn
        OV = O_dense @ evecs
        O_nn = np.einsum("ij,ij->j", np.conjugate(evecs), OV).real  # length dim

        # 6) Weighted average using scaled weights; scaling cancels
        obs = np.sum(O_nn * w) / Z_scaled

        if return_Z:
            return obs, Zpart, E0
        return obs
    #
def _calc_GS_obs(H,Os=None,return_GS = False):
    #E0, v0 = eigsh(H, k=1, which="SA")   # smallest algebraic
    #print('?')
    E0, v0 = eigsh(H, k=1, which="SA",
              ncv=80,          # try 40–200
              maxiter=200000,
              tol=1e-10)
    E0 = E0[0]
    v0 = v0[:,0]
    if return_GS is True:
        print('?',E0,v0)
        return E0,v0
    obs_out = []
    for op in Os:
        obs_out.append(np.vdot(v0, op @ v0))
    return obs_out
#
#hamiltonian
def Ham_construction_2(U,alpha,M_cut,Kappa,construct_obs = False):
    L = 2*M_cut + 1
    dim = L**2
    rows = []
    cols = []
    dat = []
    def _ind(i,j):
        return i*L + j
    #
    for i, m in enumerate(range(-M_cut, M_cut+1)):
        for j, n in enumerate(range(-M_cut, M_cut+1)):
            In = _ind(i, j)
            # diagonal
            rows.append(In); cols.append(In)
            dat.append((U/4)*(1+alpha)*m*m + (U/4)*(1-alpha)*n*n)
            #off-diagonal
            if i+1 < L and j+1 < L:
                rows.append(_ind(i+1, j+1)); cols.append(In); dat.append(0.5*Kappa)
            if i-1 >= 0 and j-1 >= 0:
                rows.append(_ind(i-1, j-1)); cols.append(In); dat.append(0.5*Kappa)
            if i+1 < L and j-1 >= 0:
                rows.append(_ind(i+1, j-1)); cols.append(In); dat.append(0.5*Kappa)
            if i-1 >= 0 and j+1 < L:
                rows.append(_ind(i-1, j+1)); cols.append(In); dat.append(0.5*Kappa)
    H = sparse.csr_matrix((dat, (rows, cols)), shape=(dim, dim))
    #################################################################################
    if construct_obs is True:
        ''' 
        constuct the cos(theta+ \pm delta theta) matrices. only have to do it once
        '''
        rows_p =[]
        cols_p =[]
        dat_p = []
        rows_m = []
        cols_m = []
        dat_m = []
        #L^2
        rows_Lsqr =[]
        cols_Lsqr =[]
        dat_Lsqr = []
        #DeltaL^2
        rows_DLsqr =[]
        cols_DLsqr =[]
        dat_DLsqr = []
        #LDeltaL
        rows_mixed =[]
        cols_mixed =[]
        dat_mixed = []

        for i, m in enumerate(range(-M_cut, M_cut+1)):
            for j, n in enumerate(range(-M_cut, M_cut+1)):
                In = _ind(i, j)
                #
                rows_DLsqr.append(In); cols_DLsqr.append(In);dat_DLsqr.append(n*n)
                rows_Lsqr.append(In); cols_Lsqr.append(In);dat_Lsqr.append(m*m)
                rows_mixed.append(In); cols_mixed.append(In);dat_mixed.append(m*n)
                #
                if i+1 < L and j+1 < L:
                    rows_p.append(_ind(i+1, j+1)); cols_p.append(In); dat_p.append(0.5)
                if i-1 >= 0 and j-1 >= 0:
                    rows_p.append(_ind(i-1, j-1)); cols_p.append(In); dat_p.append(0.5)
                #
                if i+1 < L and j-1 >= 0:
                    rows_m.append(_ind(i+1, j-1)); cols_m.append(In); dat_m.append(0.5)
                if i-1 >= 0 and j+1 < L:
                    rows_m.append(_ind(i-1, j+1)); cols_m.append(In); dat_m.append(0.5)

        Cos_p = sparse.csr_matrix((dat_p, (rows_p, cols_p)), shape=(dim, dim))
        Cos_m = sparse.csr_matrix((dat_m, (rows_m, cols_m)), shape=(dim, dim))
        Lsqr = sparse.csr_matrix((dat_Lsqr, (rows_Lsqr, cols_Lsqr)), shape=(dim, dim))
        DLsqr = sparse.csr_matrix((dat_DLsqr, (rows_DLsqr, cols_DLsqr)), shape=(dim, dim))
        LDL = sparse.csr_matrix((dat_mixed, (rows_mixed, cols_mixed)), shape=(dim, dim))
    if construct_obs is True: return H,Cos_p,Cos_m,Lsqr,DLsqr,LDL
    else: return H
#
def Ham_construction_3(U,alpha,M_cut,Kappa,construct_obs = False):
    L = 2*M_cut + 1
    dim = L**3
    rows = [];cols = [];dat = []
    def _ind(i,j,k):
        return i*L**2 + j*L + k
    # helper: add  off-diagonal element
    def _add_hop(In_from, In_to, amp):
        rows.append(In_to);   cols.append(In_from); dat.append(amp)

    # (optional) build cos(phi^eta) operators
    if construct_obs:
        r1,c1,d1 = [],[],[]
        r2,c2,d2 = [],[],[]
        r3,c3,d3 = [],[],[]

        def _add_op(r, c, d, In_from, In_to, val):
            r.append(In_to);c.append(In_from);d.append(val)

    # loop basis
    for i, m in enumerate(range(-M_cut, M_cut+1)):
        for j, n in enumerate(range(-M_cut, M_cut+1)):
            for k, ell in enumerate(range(-M_cut, M_cut+1)):
                In = _ind(i, j, k)

                # diagonal (your coefficients)
                rows.append(In); cols.append(In)
                dat.append((U/6.0)*(1+2*alpha)*m*m
                           + (U/4.0)*(1-alpha)*n*n
                           + (U/12.0)*(1-alpha)*ell*ell)

                ampH = 0.5 * Kappa  #NOTE the amplitude here....

                # --- cos(phi^1): (m,n,ell) <-> (m±1,n±1,ell±1)
                if i+1 < L and j+1 < L and k+1 < L:
                    In2 = _ind(i+1, j+1, k+1)
                    _add_hop(In, In2, ampH)
                    if construct_obs: _add_op(r1, c1, d1, In, In2,val=0.5)
                if i-1 >= 0 and j-1 >= 0 and k-1 >= 0:
                    In2 = _ind(i-1, j-1, k-1)
                    _add_hop(In, In2, ampH)
                    if construct_obs: _add_op(r1, c1, d1, In, In2,val=0.5)

                # --- cos(phi^2): (m,n,ell) <-> (m±1,n∓1,ell±1)
                if i+1 < L and j-1 >= 0 and k+1 < L:
                    In2 = _ind(i+1, j-1, k+1)
                    _add_hop(In, In2, ampH)
                    if construct_obs: _add_op(r2, c2, d2, In, In2,val=0.5)
                if i-1 >= 0 and j+1 < L and k-1 >= 0:
                    In2 = _ind(i-1, j+1, k-1)
                    _add_hop(In, In2, ampH)
                    if construct_obs: _add_op(r2, c2, d2, In, In2,val=0.5)

                # --- cos(phi^3): (m,n,ell) <-> (m±1,n,ell∓2)
                if i+1 < L and k-2 >= 0:
                    In2 = _ind(i+1, j, k-2)
                    _add_hop(In, In2, ampH)
                    if construct_obs: _add_op(r3, c3, d3, In, In2,val=0.5)
                if i-1 >= 0 and k+2 < L:
                    In2 = _ind(i-1, j, k+2)
                    _add_hop(In, In2, ampH)
                    if construct_obs: _add_op(r3, c3, d3, In, In2,val=0.5)

    H = sparse.csr_matrix((dat, (rows, cols)), shape=(dim, dim))

    if not construct_obs:
        return H

    cos1 = sparse.csr_matrix((d1, (r1, c1)), shape=(dim, dim))
    cos2 = sparse.csr_matrix((d2, (r2, c2)), shape=(dim, dim))
    cos3 = sparse.csr_matrix((d3, (r3, c3)), shape=(dim, dim))

    return H, (cos1, cos2, cos3)
#
def  Ham_construction(U,alpha,M_cut,Kappa,construct_obs = False,valleys = '2'):
    if valleys == '2':
        return Ham_construction_2(U,alpha,M_cut,Kappa,construct_obs)
    elif valleys == '3':
        return Ham_construction_3(U,alpha,M_cut,Kappa,construct_obs)
    else:
        raise ValueError(f"Unknown num={valleys}")
#
#solvers
def solver_2(U,alpha,D,beta,Z_in = 1,iterations=1000,threshold=1e-7,M_cut=10,verbose=1):
    '''
    Docstring for solver
    '''
    Zs = []
    Kappas = []
    Zs.append(Z_in)
    Kappa_in = _K_calculator(Z=Z_in,D = D,beta = beta)
    Kappas.append(Kappa_in)
    for i in range(iterations):
        time_in = time.time()
        Z_in = Zs[-1]
        Kappa_in =Kappas[-1]
        if i == 0:
            H,Cos_p,Cos_m,Lsqr,DLsqr,LDL = Ham_construction(U,alpha,M_cut,Kappa_in,construct_obs = True,valleys='2')
        else:
            H = Ham_construction(U,alpha,M_cut,Kappa_in,valleys='2')
        #
        if beta == 'inf':
            [Phi1,Phi2] = _calc_GS_obs(H,Os=[Cos_p, Cos_m],return_GS = False)
            Z_out_1 = Phi1**2
            Z_out_2 = Phi2**2
        else:
            Z_out_1 = _calc_thermal_obs(H=H,beta=beta,O=Cos_p)**2 
            Z_out_2 = _calc_thermal_obs(H=H,beta=beta,O=Cos_m)**2
        if np.abs(Z_out_1 - Z_out_2) > threshold:
            raise ValueError(f"Symmetry broken: {Z_out_1:.3f} =Z_plus != Z_minus={Z_out_2:.3f}")
        diff = np.abs(Z_out_1-Z_in)
        if verbose > 0:print(f'Step:{i}|KappaZ:{diff:.3e}|time:{time.time()-time_in:.2f}')
        if np.abs(diff) < threshold:
            if verbose > 0:print(f'Converged early at {i} iteration')
            return  Zs,Kappas      
        Zs.append(Z_out_1)
        Kappas.append(_K_calculator(Z=Z_out_1,D = D,beta = beta))
    return Zs,Kappas  
def solver_3(U,alpha,D,beta,Z_in = 1,iterations=1000,threshold=1e-7,M_cut=10,verbose = 1):
    '''
    Docstring for solver
    '''
    Zs = []
    Kappas = []
    Zs.append(Z_in)
    Kappa_in = _K_calculator(Z=Z_in,D = D,beta = beta)
    Kappas.append(Kappa_in)
    for i in range(iterations):
        time_in = time.time()
        Z_in = Zs[-1]
        Kappa_in = Kappas[-1]
        if i == 0:
            H,(Cos_1, Cos_2, Cos_3) = Ham_construction(U,alpha,M_cut,Kappa = Kappa_in,construct_obs = True,valleys='3')
        else:
            H = Ham_construction(U,alpha,M_cut,Kappa = Kappa_in,valleys='3')
        if beta == 'inf':
            [Phi1,Phi2,Phi3] = _calc_GS_obs(H,Os=[Cos_1, Cos_2, Cos_3],return_GS = False)
            Z_out_1 = Phi1**2
            Z_out_2 = Phi2**2
            Z_out_3 = Phi3**2
        else:
            Z_out_1 = _calc_thermal_obs(H=H,beta=beta,O=Cos_1)**2 
            Z_out_2 = _calc_thermal_obs(H=H,beta=beta,O=Cos_2)**2
            Z_out_3 = _calc_thermal_obs(H=H,beta=beta,O=Cos_3)**2 
        if (np.abs(Z_out_1 - Z_out_2) > threshold) or (np.abs(Z_out_1 - Z_out_3) > threshold):
            if verbose > 0:print(f"Symmetry broken:Z1,Z2,Z3: {Z_out_1:.3f},{Z_out_2:.3f},{Z_out_3:.3f}")
        diff = np.abs(Z_out_1-Z_in)
        if verbose > 0:print(f'Step:{i}|DeltaZ:{diff:.3e}|time:{time.time()-time_in:.2f}')
        if np.abs(diff) < threshold:
            if verbose > 0:print(f'Converged early at {i} iteration')
            return  Zs,Kappas      
        Zs.append(Z_out_1)
        Kappas.append(_K_calculator(Z=Z_out_1,D = D,beta=beta))
    return Zs,Kappas
def solver(U,alpha,D,beta,Z_in = 1,iterations=1000,threshold=1e-7,M_cut=10, valleys='2',verbose = 1):
    if valleys == '2':
        return solver_2(U,alpha,D,beta,Z_in = Z_in,iterations=iterations,threshold=threshold,M_cut=M_cut,verbose = verbose)
    elif valleys == '3':
        return solver_3(U,alpha,D,beta,Z_in = Z_in,iterations=iterations,threshold=threshold,M_cut=M_cut,verbose = verbose)
    else:
        raise ValueError(f"Unknown num={valleys}")
#
def critical_U_finder(alpha,U_L,U_R,D = 1,beta = 'inf',M_cut = 10,Z_in = 1,zero=1e-3,accuracy=0.01,valleys = '3'):
    ''' 
    Finds U critical for a given alpha, within *acuuracy*
    '''
    iterations = int(np.log((U_R-U_L)/accuracy)/np.log(2))
    print(f'iterations:{iterations}')
    for _ in range(iterations):
        Z_L = solver(U=U_L,alpha=alpha,D = D, beta=beta,Z_in = Z_in,M_cut=M_cut,valleys=valleys,verbose=0)[0][-1]
        Z_R = solver(U=U_R,alpha=alpha,D = D, beta=beta,Z_in = Z_in,M_cut=M_cut,valleys=valleys,verbose=0)[0][-1]
        if (Z_L < zero ) or (Z_R > zero):
            print('uhhh?')
            return (U_L+U_R)/2
        U_mid = 0.5*(U_L + U_R)
        Z_mid = solver(U=U_mid,alpha=alpha,D = D, beta=beta,Z_in = Z_in,M_cut=M_cut,valleys=valleys,verbose=0)[0][-1]
        if Z_mid > zero:
            U_L = U_mid
        elif Z_mid <= zero:
            U_R = U_mid
        print(f'updated range:[{U_L,U_R}]')
    return (U_L+U_R)/2

#####################################################
def Z_vs_U_vs_alpha():
    alphas = np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1.])
    print('alphas',alphas)
    beta = 'inf'
    Us = np.linspace(0,10,num=30)
    data_out = {}
    for alpha in alphas:
        for u in Us:
            Zs,Kappas = solver(U=u,alpha=alpha,D = 2.79, beta=beta,Z_in = 1,M_cut=15,valleys='3')
            data_out[(u,alpha)] = (Zs[-1],Kappas[-1])
    return data_out
def U_critical(U_L,U_R):
    alphas = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.985,0.9875,0.99,0.9925,0.9950,0.9975,1])
    U_c_alphas = {}
    accuracy = 0.05
    for alpha in alphas:
        U_c = critical_U_finder(alpha = alpha,U_L = U_L,U_R = U_R,D = 2.79,beta = 'inf',M_cut = 15,Z_in = 1,zero=1e-3,accuracy=accuracy,valleys = '3')
        U_c_alphas[alpha] = (U_c,accuracy)
    return U_c_alphas
###########################################################
if __name__ == "__main__":
    #data = Z_vs_U_vs_alpha()
    data = U_critical(U_L=2,U_R=9)
    with open("data_Uc.pkl", "wb") as f:
        pickle.dump(data, f)
    