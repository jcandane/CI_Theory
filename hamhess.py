import numpy as np 

π = np.pi 

def get_H1_AO(mf):
    """
    GIVEN:  mf (PySCF Mean-Field (MF) Object, e.g. HF/KS)
    GET:    1-body AO Hamiltonian
    """
    H_ao = mf.get_hcore()
    try: ## if KS, compute V_XC
        mf.xc
        ni = dft.numint.NumInt()
        nelec, exc, vxc = ni.nr_uks(mf.mol, mf.grids, mf.xc, mf.make_rdm1())
        if vxc.ndim==3:
            H_ao  = vxc + H_ao[None,:,:]
        else:
            H_ao += vxc
    except:
        None

    return H_ao

def get_H2_AO(mf):
    """
    GIVEN:  mf (PySCF Mean-Field (MF) Object, e.g. HF/KS)
    GET:    2-body AO Hamiltonian
    """
    return mf._eri

def get_MO_H1(scf, ξ=None, GAS="AA", supp=None):
    """
    GIVEN:  mf (PySCF Mean-Field (MF) Object, e.g. HF/KS)
            AO-Hamiltonian
            ξ (GAS vector, Generalized Active Space, active MOs)
            GAS (GAS String)
    GET:    MO-Hamiltonian
    Idea: from mf -> AO Hamiltonian, MO-coefficients 
          -> dress AO Ham with MO coefficients
    """
    ## get AO Hamiltonian
    H_ao = get_H1_AO(scf)

    if supp is not None:
        H_ao += supp

    ## get MO Coefficients
    if scf.mo_coeff.ndim == 3:
        Ca, Cb = (scf).mo_coeff
        C      = (Ca + Cb) / 2
    else:
        C = (scf).mo_coeff

    ## Implement GAS-vector
    if ξ is None:
          ξ  = np.arange(0, len(Ca), 1, dtype=int)
    o = np.arange(0, len(Ca), 1, dtype=int)
    χ = np.delete(o, ξ)

    if GAS[0] == "I":
        a = χ
    else:
        a = ξ
    if GAS[1] == "I":
        b = χ
    else:
        b = ξ

    ## MO Transformation
    if H_ao.ndim == 3:
        return np.array([ C[:,a].T @ H_ao[0] @ C[:,b], C[:,a].T @ H_ao[1] @ C[:,b] ])
    else:
        return C[:,a].T @ H_ao @ C[:,b]

def get_MO_H2(scf, ξ=None, GAS="AAAA"):
    """
    GIVEN:  mf (PySCF Mean-Field (MF) Object, e.g. HF/KS)
            AO-Hamiltonian
            ξ (GAS vector, Generalized Active Space, active MOs)
            GAS (GAS String)
    GET:    SO-Hamiltonian
    Idea: from mf -> AO Hamiltonian (ERI), MO-coefficients 
          -> dress AO Ham with MO coefficients
    """
    ## get MO Coefficients
    if scf.mo_coeff.ndim == 3:
        Ca, Cb = (scf).mo_coeff
        C      = (Ca + Cb) / 2
    else:
        C = (scf).mo_coeff
    
    ## Implement GAS-vector
    if ξ is None:
        ξ  = np.arange(0, C.shape[-1], 1, dtype=int)
    o = np.arange(0, C.shape[-1], 1, dtype=int)
    χ = np.delete(o, ξ)

    if GAS[0] == "I":
        a = χ
    else:
        a = ξ
    if GAS[1] == "I":
        b = χ
    else:
        b = ξ
    if GAS[2] == "I":
        c = χ
    else:
        c = ξ
    if GAS[3] == "I":
        d = χ
    else:
        d = ξ

    ## MO Transformation
    HF_EXCHANGE = True
    try:
        scf.xc
        HF_EXCHANGE = False
    except:
        None
    aa = a.shape[0]
    bb = b.shape[0]
    cc = c.shape[0]
    dd = d.shape[0]
    eri_aa = (ao2mo.general( get_H2_AO(scf) , (C[:,a], C[:,b], C[:,c], C[:,d]), compact=False)).reshape((aa,bb,cc,dd), order="C")
    if np.array_equal(a, b) and np.array_equal(c, d) and np.array_equal(a, d) and HF_EXCHANGE:
        eri_aa -= eri_aa.swapaxes(1,3)
    return eri_aa

def get_SO_H1(scf, ξ=None, GAS="AA", supp=None):
    """
    GIVEN:  mf (PySCF Mean-Field (MF) Object, e.g. HF/KS)
            AO-Hamiltonian
            ξ (GAS vector, Generalized Active Space, active MOs)
            GAS (GAS String)
    GET:    SO-Hamiltonian
    Idea: from mf -> AO Hamiltonian, MO-coefficients 
          -> dress AO Ham with MO coefficients
    """
    ## get AO Hamiltonian
    H_ao = get_H1_AO(scf)
    if H_ao.ndim == 2:
        H_ao = np.array([H_ao, H_ao])

    ## add KS-DFT part if needed
    try:
        scf.xc
        ni = dft.numint.NumInt()
        nelec, exc, vxc = ni.nr_uks(scf.mol, scf.grids, scf.xc, scf.make_rdm1())
        H_ao += vxc
    except:
        None

    if supp is not None:
        H_ao += supp

    ## get MO Coefficients
    Ca, Cb = (scf).mo_coeff
    if ξ is None:
          ξ  = np.arange(0, len(Ca), 1, dtype=int)
    o = np.arange(0, len(Ca), 1, dtype=int)
    χ = np.delete(o, ξ)

    if GAS[0] == "I":
        a = χ
    else:
        a = ξ
    if GAS[1] == "I":
        b = χ
    else:
        b = ξ

    return np.array([ Ca[:,a].T @ H_ao[0] @ Ca[:,b] , Cb[:,a].T @ H_ao[0] @ Cb[:,b] ])

def get_SO_H2(scf, ξ=None, GAS="AAAA"):
    """
    GIVEN:  mf (PySCF Mean-Field (MF) Object, e.g. HF/KS)
            AO-Hamiltonian
            ξ (GAS vector, Generalized Active Space, active MOs)
            GAS (GAS String)
    GET:    SO-Hamiltonian
    Idea: from mf -> AO Hamiltonian (ERI), MO-coefficients 
          -> dress AO Ham with MO coefficients
    """
    ## get MO Coefficients
    Ca, Cb = (scf).mo_coeff
    
    if ξ is None:
        ξ  = np.arange(0, len(Ca), 1, dtype=int)
    o = np.arange(0, len(Ca), 1, dtype=int)
    χ = np.delete(o, ξ)

    if GAS[0] == "I":
        a = χ
    else:
        a = ξ
    if GAS[1] == "I":
        b = χ
    else:
        b = ξ
    if GAS[2] == "I":
        c = χ
    else:
        c = ξ
    if GAS[3] == "I":
        d = χ
    else:
        d = ξ

    HF_EXCHANGE = True
    try:
        scf.xc
        HF_EXCHANGE = False
    except:
        None
    
    aa = a.shape[0]
    bb = b.shape[0]
    cc = c.shape[0]
    dd = d.shape[0]
    eri_aa = (ao2mo.general( get_H2_AO(scf) , (Ca[:,a], Ca[:,b], Ca[:,c], Ca[:,d]), compact=False)).reshape((aa,bb,cc,dd), order="C")
    eri_bb = (ao2mo.general( get_H2_AO(scf) , (Cb[:,a], Cb[:,b], Cb[:,c], Cb[:,d]), compact=False)).reshape((aa,bb,cc,dd), order="C")
    if np.array_equal(a, b) and np.array_equal(c, d) and np.array_equal(a, d) and HF_EXCHANGE:
        eri_aa -= eri_aa.swapaxes(1,3)
        eri_bb -= eri_bb.swapaxes(1,3)
    eri_ab = (ao2mo.general( get_H2_AO(scf) , (Ca[:,a], Ca[:,b], Cb[:,c], Cb[:,d]), compact=False)).reshape((aa,bb,cc,dd), order="C")
    #if np.array_equal(a, b) and np.array_equal(c, d) and np.array_equal(a, d) and HF_EXCHANGE:
    #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry
    eri_ba = (ao2mo.general( get_H2_AO(scf) , (Cb[:,a], Cb[:,b], Ca[:,c], Ca[:,d]), compact=False)).reshape((aa,bb,cc,dd), order="C")
    return np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

def get_CI_H1(H1_SO, SC1, Binary):
    I_A, J_A, a, a_t, I_B, J_B, b, b_t = SC1

    H1_CI   = np.diag( np.einsum("spp, sIp -> I", H1_SO, Binary) )
    H1_CI[I_A, J_A] += np.einsum("pq, Kp, Kq -> K", H1_SO[0], a_t, a)
    H1_CI[I_B, J_B] += np.einsum("pq, Kp, Kq -> K", H1_SO[1], b_t, b)

    return H1_CI

def get_CI_H2(H2_SO, SC1, SC2, Binary):
    I_A, J_A, a, a_t, I_B, J_B, b, b_t = SC1
    I_AA, J_AA, aa_T, I_AB, J_AB, ab_T, I_BB, J_BB, bb_T, ca, cb = SC2

    H2_CI    = np.diag( np.einsum("stppqq, sIp, tIq -> I", H2_SO, Binary, Binary, optimize=True)/2 )
    H2_CI[I_A, J_A]   = np.einsum("sprqq, Kp, Kr, sKq -> K", H2_SO[0], a_t, a, ca, optimize=True)
    H2_CI[I_B, J_B]   = np.einsum("sprqq, Kp, Kr, sKq -> K", H2_SO[1], b_t, b, cb, optimize=True)
    H2_CI[I_AA, J_AA] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2_SO[0,0], aa_T[0], aa_T[1], aa_T[2], aa_T[3], optimize=True)
    H2_CI[I_BB, J_BB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2_SO[1,1], bb_T[0], bb_T[1], bb_T[2], bb_T[3], optimize=True)
    H2_CI[I_AB, J_AB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2_SO[0,1], ab_T[0], ab_T[1], ab_T[2], ab_T[3], optimize=True)

    return H2_CI


