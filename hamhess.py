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


