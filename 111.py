import os
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# Read the Input files and parameters
path_to_files = "./ints/"

S = np.load(path_to_files + "S.npy")
V = np.load(path_to_files + "V.npy")
E_nuc = np.load(path_to_files + "enuc.npy")
h = np.load(path_to_files + "h.npy")

print("Shape of S:", S.shape)   # (7,7) for water
print("Shape of V:", V.shape)  # (7, 7, 7, 7) for water (N_{AO} × N_{AO} × N_{AO} × N_{AO})
print("Shape of enuc:", E_nuc.shape)
print("Shape of h:", h.shape)  # (7,7) for water (\mu, \nu)

# Number of electrons
n_electrons = 10

# Initial Guess
def initial_guess(S, h, n_electrons):
    epsilon, C = eigh(h, S)
    n_occ = n_electrons // 2

    # Define the density matrix $ D_{\mu \nu} ^{(0)} = 2 \sum_i^{0} C_{\mu i} ^{(0)} C_{\nu i} ^{(0)}  $
    D = np.zeros((C.shape[0], C.shape[0]))
    for i in range(n_occ):
        D += 2 * np.dot(C[:, i:i+1], C[:, i:i+1].T)
    return C, D, epsilon, n_occ

# Compute the Fock Matrix
# $F_{\mu \nu}^{(0)} = h_{\mu \nu} + \sum_{\lambda \sigma} [2 (\mu \nu | \lambda \sigma) - (\mu \sigma | \lambda \nu ) ] D_{\sigma \lambda}^{(0)}$ 
def compute_fock_matrix(h, ERIs, D):
    F = h.copy()
    for mu in range(F.shape[0]):
        for nu in range(F.shape[1]):
            # Coulomb and exchange contributions
            J = np.sum(ERIs[mu, nu, :, :] * D)
            K = np.sum(ERIs[mu, :, :, nu] * D)
            F[mu, nu] += J -  0.5* K
    return F

# Compute the total energy
# $E_{tot}^{(0)} = E_{nuc} + Tr D^{(0)} (h + F^{(0)})$
def compute_total_energy(h, F, D, E_nuc):
    # E_tot = E_nuc + 0.5 * np.sum(D* (h + F))
    E_tot = E_nuc + 0.5 * np.trace(np.dot(D, h + F))
    return E_tot

def direct_inversion(F_list, cycle, error_list):
    print(f"Cycle value is: {cycle}, its type is: {type(cycle)}")
    size = len(F_list) + 1
    # size = int(cycle)+int(1)

    B = np.zeros((size, size))
    B[ : , -1] = -1
    B[-1, : ] = -1
    B[-1, -1] = 0


    for i in range(cycle):
        for j in range(cycle):
            B[i,j] = np.dot(error_list[i].ravel(), error_list[j].ravel())

    rhs = np.zeros(size)
    rhs[-1] = -1

    c_vector = np.linalg.solve(B, rhs)

    #c = c_vector[8:-1]
    c = c_vector[:-1]
    c = np.array(c).reshape(-1, 1, 1)

    #F_array = np.array(F_list[8:])
    F_array = np.array(F_list)
    F_weighted = np.sum(F_array * c, axis=0)


    return F_weighted
        

def MP2_correct(ERI_AO, C, epsilon, n_occ):
    ERI_MO = np.einsum('pqrs, pi, qj, rk, sl -> ijkl', ERI_AO, C, C, C, C)
    n_uno = epsilon.shape[0] - n_occ
    E_MP2 = 0.0
    for i in range(n_occ):
        for j in range(n_occ):
            for a in range(n_occ, epsilon.shape[0]):
                for b in range(n_occ, n_occ + n_uno):
                    iajb = ERI_MO[i, a, j, b]
                    ibja = ERI_MO[i, b, j, a]
                    E_MP2 += iajb * (2* iajb - ibja) / (epsilon[i] + epsilon[j] - epsilon[a] - epsilon[b])
    
    return E_MP2
        
    


def scf_solver(S, h, ERIs, E_nuc, beta, max_cycles=100, tol=1e-8, tol2=1e-4):
    C, D, epsilon, n_occ = initial_guess(S, h, n_electrons)
    E_old = 0
    D_old = D.copy()
    converged = False
    energy_diffs = []
    E_tots = []

    F_list = []
    error_list = []

    for cycle in range(max_cycles):
        print(type(cycle))
        D = ( 1- beta) * D + beta * D_old
        
        F = compute_fock_matrix(h, ERIs, D)
        F_list.append(F)


        C, D_new, epsilon, n_occ = initial_guess(S, F, n_electrons)

        commutator = np.dot(F, np.dot(D, S)) - np.dot(S, np.dot(D, F))
        error_list.append(commutator)
        if cycle > 8:
            F_ave = direct_inversion(F_list, cycle, error_list)
        else:
            F_ave = F


        E_tot = compute_total_energy(h, F_ave, D_new, E_nuc)

        energy_diff = abs(E_tot - E_old)
        # || D^{(n)} - D^{(n-1)} ||_F < \sigma_{dm}  Frobenius norm
        density_diff = np.linalg.norm(D - D_new, ord='fro')

        print('commutator', commutator.shape)
        commutator_diff = np.linalg.norm(commutator, ord='fro')

        print(f"Iteration {cycle + 1}")
        print(f"Energy Difference: {energy_diff:.8f}")
        print(f"Density Matrix Difference: {density_diff:.8f}")
        print(f"Commutator Difference: {commutator_diff:.8f}")

        energy_diffs.append(energy_diff)
        E_tots.append(E_tot)

        #if (energy_diff < tol) and (density_diff < tol2) and (commutator_diff < tol2):
        if energy_diff < tol:
            converged = True
            print("reached required accuracy - stopping structural energy minimisation.")
            E_MP2 = MP2_correct(V, C, epsilon, n_occ)
            print("The correction of MP2 method is ", E_MP2)
            E_final = E_tot + E_MP2
            print("The final energy is ", E_final)
            break

        E_old = E_tot
        D = D_new
    
    if not converged:
        print("Max iterations reached without convergence.")

    return E_tot, D, energy_diffs, E_tots

# Run the SCF solver
E_tot, D_final, energy_diffs, E_tots = scf_solver(S, h, V, E_nuc,beta=0)
print("Final Total Energy:", E_tot)

# Draw the figure of energy_diffs and E_tots
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(energy_diffs, marker = 'o')
plt.xlabel('scf cycle')
plt.ylabel('Energy Difference (Hartree)')
plt.title('Energy Difference vs scf cycle')
plt.xticks(np.arange(1, len(energy_diffs) + 1, 2))  

plt.subplot(1, 2, 2)
plt.plot(E_tots, marker = 'o')
plt.xlabel('scf cycle')
plt.ylabel('Energy (Hartree)')
plt.title('Energy vs scf cycle')
plt.xticks(np.arange(1, len(E_tots) + 1, 2))  

plt.tight_layout()
plt.savefig('hf.png',dpi=300)
plt.show()