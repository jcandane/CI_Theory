import numpy as np 
from itertools import combinations

π = np.pi 

def Ext(A, N, numtype=np.int8):
    if A.ndim == 1:
        return np.einsum("I, j -> Ij", A, np.ones( N, dtype=numtype))
    if A.ndim == 2:
        return np.einsum("uI, j -> uIj", A, np.ones( N, dtype=numtype))

def ΛMOgetB(Λ, N_mo):
  "Given Λ (i occupied orbitals for each determinant) get B (binary rep.)"

  Binary  = np.zeros((Λ.shape[0], N_mo), dtype=np.int8)
  for I in range(len(Binary)):
      Binary[I, Λ[I,:]] = 1

  return Binary

def SpinOuterProduct(A, B, stack=False):
  ΛA = np.einsum("Ii, J -> IJi", A, np.ones(B.shape[0], dtype=np.int8)).reshape( (A.shape[0]*B.shape[0], A.shape[1]) )
  ΛB = np.einsum("Ii, J -> JIi", B, np.ones(A.shape[0], dtype=np.int8)).reshape( (A.shape[0]*B.shape[0], B.shape[1]) )
  
  if stack:
    return np.array([ΛA,ΛB])
  else:
    return ΛA, ΛB

def CIunique(dets):
    BB = np.unique( ( dets.swapaxes(0,1)).reshape(dets.shape[1], dets.shape[0]*dets.shape[2], order="C") , axis=0)
    return BB.reshape(BB.shape[0], 2, BB.shape[1]//2, order="C").swapaxes(0,1)

def combine_arrays(Λ_sample, V_sample):

    if V_sample.size == 0:
        return Λ_sample

    else:
        l = V_sample.shape[1] + Λ_sample.shape[1] # length of new array
        m = V_sample.shape[1] # level of excitation
        a_new = np.empty((V_sample.shape[0] * Λ_sample.shape[0], V_sample.shape[1] + Λ_sample.shape[1] ), dtype=np.int8)
        a_new[:,:l - m ] = np.repeat(Λ_sample, len(V_sample), axis=0)
        a_new[:, l - m:] = np.repeat(V_sample, len(Λ_sample), axis=1).reshape((len(Λ_sample)*len(V_sample), V_sample.shape[1]), order="F")

        return np.sort(a_new, axis=1)

def MO_X(MO_p, m, SF=False):
    """ 
    m-level CI excitations on an MO 
    m=0 should return the reference
    """

    occ = np.asarray([ np.where(MO_p == 1)[0] ]) ## occupied space
    vac = np.asarray([ np.where(MO_p == 0)[0] ]) ## vacant   space

    if SF: ### do SF-calculation
        if m < 0 and abs(m) <= len(occ[0]): ## if m is negative, then we do all combinations deleting occupied orbital(s) 
            Occ_choices = np.asarray( list(combinations(  occ[0], len(occ[0]) + m ) ) , dtype=np.int8)
            return ΛMOgetB(Occ_choices, len(MO_p))
        if m == 0: ## if m = 0 , return reference
            return np.asarray([MO_p])
        if m > 0 and m <= len(vac[0]): ## if m is positive, we will do all combinations by adding occupied orbital(s) 
            Vac_choices = np.asarray( list(combinations(  vac[0], m ) ) , dtype=np.int8)
            return ΛMOgetB(combine_arrays(occ, Vac_choices), len(MO_p))
        else:
            return None

    else: ### MO-conserved-calculation
        ## excitation order must be less than avaiable orbitals
        if m <= len(occ[0]) and m <= len(vac[0]): 
            Occ_choices = np.asarray( list(combinations(  occ[0], len(occ[0]) - m ) ) , dtype=np.int8)
            Vac_choices = np.asarray( list(combinations(  vac[0], m ) ) , dtype=np.int8)
            return ΛMOgetB(combine_arrays(Occ_choices, Vac_choices), len(MO_p))
        else:
            return None

def SO_X(ref_SO, m, SF=False):

    if SF:
        C = None
        D = None
        ## +3 & -3
        A = MO_X(ref_SO[0],  m, SF=True)
        B = MO_X(ref_SO[1], -m, SF=True)
        if A is not None and B is not None:
            C = SpinOuterProduct(A, B, stack=True)
        
        ## -3 & +3
        A = MO_X(ref_SO[0], -m, SF=True)
        B = MO_X(ref_SO[1],  m, SF=True)
        if A is not None and B is not None:
            D = SpinOuterProduct(A, B, stack=True)
        
        if C is not None and D is not None:
            return np.append(D, C, axis=1)
        if C is not None:
            return C
        if D is not None:
            return D
        else:
            return None

    else:
        D     = None
        first = True
        for j in range(m+1):
            A = MO_X(ref_SO[0],   j)
            B = MO_X(ref_SO[1], m-j)
            if A is not None and B is not None:
                C = SpinOuterProduct(A, B, stack=True)
                if first:
                    D = C
                    first = False
                else:
                    D = np.append(D, C, axis=1)
            else:
                continue
        return D

def Ref_X(ref_SO, RAS=None, CIm=None, SFm=None):

    if RAS is None:
        RAS = np.arange(0, ref_SO.shape[1], 1, dtype=int)

    RASref = ref_SO[:, RAS]
    #loop   = Ext(reference, RASCI.shape[1]).swapaxes(1,2)
    

    ### CIm Loop
    outCII = None
    if CIm is not None:
        first = True
        for i in range(len(CIm)):
            CIm_sIp = SO_X(RASref, CIm[i], SF=False)
            if first and CIm_sIp is not None:
                outCI = CIm_sIp
                first = False
                continue
            if CIm_sIp is not None:
                outCI = np.append(outCI, CIm_sIp, axis=1)
            else:
                continue
        ## refit with RAS 
        outCII = Ext(ref_SO, outCI.shape[1]).swapaxes(1,2)
        outCII[:,:,RAS] = outCI

    ### SFm Loop
    outSFF = None
    if SFm is not None:
        first = True
        for i in range(len(SFm)):
            SFm_sIp = SO_X(RASref, SFm[i], SF=True)
            if first and SFm_sIp is not None:
                outSF = SFm_sIp
                first = False
                continue
            if SFm_sIp is not None:
                outSF = np.append(outSF, SFm_sIp, axis=1)
            else:
                continue
        ## refit with RAS 
        outSFF = Ext(ref_SO, outSF.shape[1]).swapaxes(1,2)
        outSFF[:,:,RAS] = outSF

    return outCII, outSFF

def MR_X(MR_SO, RAS=None, CIm=None, SFm=None):

    firstCI = True
    firstSF = True
    outputCI = None
    outputSF = None

    if RAS is not None: ## checking RAS
        if np.max(RAS) > MR_SO.shape[-1]:
            RAS = None

    if MR_SO.ndim == 2:
        outCI, outSF = Ref_X(MR_SO, RAS=RAS, CIm=CIm, SFm=SFm)

        if outCI is not None and firstCI:
            outputCI = outCI
            firstCI  = False
        if outCI is not None:
            outputCI = np.append(outputCI, outCI, axis=1)
            outputCI = CIunique(outputCI)
        if outSF is not None and firstSF:
            outputSF = outSF
            firstSF  = False
        if outSF is not None:
            outputSF = np.append(outputSF, outSF, axis=1)
            outputSF = CIunique(outputSF)

    if MR_SO.ndim == 3:

        for i in range(len(MR_SO)):
            outCI, outSF = Ref_X(MR_SO[i], RAS=RAS, CIm=CIm, SFm=SFm)

            if outCI is not None and firstCI:
                outputCI = outCI
                firstCI  = False
            if outCI is not None:
                outputCI = np.append(outputCI, outCI, axis=1)
                outputCI = CIunique(outputCI)
            if outSF is not None and firstSF:
                outputSF = outSF
                firstSF  = False
            if outSF is not None:
                outputSF = np.append(outputSF, outSF, axis=1)
                outputSF = CIunique(outputSF)

    return outputCI, outputSF


