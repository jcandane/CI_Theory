import numpy as np 

π = np.pi 
CI_dt  = np.int32

def givenBgetΛ(B):
  "Given B (entire MO binary occupation) get Λ (i occupied)"

  numA = len( (B[0,0])[ B[0,0] == 1 ] ) ## count num. of A occupied
  numB = len( (B[0,1])[ B[0,1] == 1 ] ) ## count num. of B occupied

  ΛA = np.zeros((B.shape[0], numA), dtype=np.int8)
  ΛB = np.zeros((B.shape[0], numB), dtype=np.int8)
  for I in range(len(B)):
    ΛA[I] = np.where(B[I,0] == 1)[0]
    ΛB[I] = np.where(B[I,1] == 1)[0]

  return ΛA, ΛB

def SlaterCondon(Binary):
    N_s    = np.einsum("sp -> s", Binary[0])
    Difference     = np.einsum("Isp, J -> IJsp", Binary, np.ones(len(Binary), dtype=np.int8)) - np.einsum("Isp, J -> JIsp", Binary, np.ones(len(Binary), dtype=np.int8))
    Sum            = np.einsum("Isp, J -> IJsp", Binary, np.ones(len(Binary), dtype=np.int8)) + np.einsum("Isp, J -> JIsp", Binary, np.ones(len(Binary), dtype=np.int8))
    SpinDifference = np.einsum("IJsp -> IJs", np.abs(Difference))//2

    ##  indices for 1-difference
    I_A, J_A = np.where( np.all(SpinDifference==np.array([1,0], dtype=CI_dt), axis=2) )
    I_B, J_B = np.where( np.all(SpinDifference==np.array([0,1], dtype=CI_dt), axis=2) )
    I_A = I_A.astype(CI_dt)
    J_A = J_A.astype(CI_dt)
    I_B = I_B.astype(CI_dt)
    J_B = J_B.astype(CI_dt)

    A_t  = (np.where( Difference[I_A, J_A, 0] ==  1 )[1]).astype(CI_dt)
    A    = (np.where( Difference[I_A, J_A, 0] == -1 )[1]).astype(CI_dt)
    B_t  = (np.where( Difference[I_B, J_B, 1] ==  1 )[1]).astype(CI_dt)
    B    = (np.where( Difference[I_B, J_B, 1] == -1 )[1]).astype(CI_dt)
    CA_i = ((np.where( Sum[I_A, J_A, 0] == 2 )[1]).reshape(len(I_A), N_s[0]-1)).astype(CI_dt)
    CB_i = ((np.where( Sum[I_B, J_B, 1] == 2 )[1]).reshape(len(I_B), N_s[1]-1)).astype(CI_dt)

    ## indices for 2-differences
    I_AA, J_AA = np.where( np.all(SpinDifference==np.array([2,0], dtype=CI_dt), axis=2) )
    I_BB, J_BB = np.where( np.all(SpinDifference==np.array([0,2], dtype=CI_dt), axis=2) )
    I_AB, J_AB = np.where( np.all(SpinDifference==np.array([1,1], dtype=CI_dt), axis=2) )
    I_AA = I_AA.astype(CI_dt)
    J_AA = J_AA.astype(CI_dt)
    I_BB = I_BB.astype(CI_dt)
    J_BB = J_BB.astype(CI_dt)
    I_AB = I_AB.astype(CI_dt)
    J_AB = J_AB.astype(CI_dt)

    AA   = (np.where( Difference[I_AA, J_AA, 0] == -1)[1].reshape(len(I_AA),2).T).astype(CI_dt)
    AA_t = (np.where( Difference[I_AA, J_AA, 0] ==  1)[1].reshape(len(I_AA),2).T).astype(CI_dt)
    BB   = (np.where( Difference[I_BB, J_BB, 1] == -1)[1].reshape(len(I_BB),2).T).astype(CI_dt)
    BB_t = (np.where( Difference[I_BB, J_BB, 1] ==  1)[1].reshape(len(I_BB),2).T).astype(CI_dt)
    AB   = np.asarray([ np.where( Difference[I_AB, J_AB, 0] == -1 )[1], np.where( Difference[I_AB, J_AB, 1] == -1 )[1] ], dtype=CI_dt)
    AB_t = np.asarray([ np.where( Difference[I_AB, J_AB, 0] ==  1 )[1], np.where( Difference[I_AB, J_AB, 1] ==  1 )[1] ], dtype=CI_dt)

    ## get orbital occupations for each up-xor-down
    ΛA, ΛB = givenBgetΛ(Binary)

    ## get sign
    sign  = np.cumsum( Binary, axis=2)
    for I in range(len(Binary)):
        sign[I, 0, ΛA[I]] = np.arange(0, N_s[0], 1)
        sign[I, 1, ΛB[I]] = np.arange(0, N_s[1], 1)

    Γ_Isp = ( (-1)**(sign) ).astype(np.int8)

    return [I_A, J_A, A, A_t, I_B, J_B, B, B_t, CA_i, CB_i], [I_AA, J_AA, AA, AA_t, I_AB, J_AB, AB, AB_t, I_BB, J_BB, BB, BB_t], Γ_Isp


