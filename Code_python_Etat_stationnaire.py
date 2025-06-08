import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# TRANSFORMATION DES UNITÉS : définition de la conversion
# =====================================================================
hbar     = 1.054571817e-34     # J·s
mass_He  = 6.646476e-27        # kg (masse d’un atome de 4He)
e_charge = 1.602176634e-19     # C = charge élémentaire (J/eV)
mu       = mass_He / 2         # masse réduite pour deux atomes identiques
a_real   = 3e-10               # m  (demi-largeur "réelle")
E_unit   = (hbar**2) / (2 * mass_He * a_real**2)  # J par unité-modèle

# =====================================================================
# PARAMÈTRES DU PUITS (unités-modèle)
# =====================================================================
V0_model  = 20.0   # profondeur du puits
a_model   = 1.0    # demi-largeur
n_vals    = np.arange(1, 6)
E_n_model = (n_vals**2 * np.pi**2) / (4 * a_model**2) - V0_model
E_n_real  = E_n_model * E_unit
E_n_eV    = E_n_real / e_charge

# Affichage des énergies analytiques
print("Énergies analytiques (n=1..5) :")
for n, Em, Ej, Ee in zip(n_vals, E_n_model, E_n_real, E_n_eV):
    tag = "continuum" if Em>0 else "lié"
    print(f"  n={n}: E_mod={Em:.4f} → E={Ej:.3e} J ≃ {Ee:.3e} eV ({tag})")

# ---------------------------------------------------------------------
# 1) Fonctions pour diffusion (E>0) : calcul de r, B, C, D
# ---------------------------------------------------------------------
def solve_coefficients(E_mod, V0, a):
        # k    : onde libre hors du puits  (E = k²)
    # q    : onde dans le puits       (E + V0 = q²)
    k = np.sqrt(E_mod + 0j)
    q = np.sqrt(E_mod + V0 + 0j)

      # Exponentielles pour alléger les notations
    e_mika = np.exp(-1j * k * a)
    e_pika = np.exp( 1j * k * a)
    e_miqa = np.exp(-1j * q * a)
    e_piqa = np.exp( 1j * q * a)

    A    = np.zeros((4,4), dtype=complex)
    Bvec = np.zeros(4,    dtype=complex)

    # Continuité en x = -a
    A[0]   = [ e_pika,   -e_miqa,   -e_piqa,   0    ]
    Bvec[0]= -e_mika
      # Continuité de ψ' à x=-a
    A[1]   = [-1j*k*e_pika, -1j*q*e_miqa, +1j*q*e_piqa, 0]
    Bvec[1]= -1j*k*e_mika

    # Continuité en x = +a
    A[2]   = [0, e_piqa, e_miqa, -e_pika]
    Bvec[2]= 0
      # Continuité de ψ' à x=+a
    A[3]   = [0, 1j*q*e_piqa, -1j*q*e_miqa, -1j*k*e_pika]
    Bvec[3]= 0

    sol = np.linalg.solve(A, Bvec)
    return sol[0], sol[1], sol[2], sol[3]  # r, B, C, D

# ---------------------------------------------------------------------
# 2) Fonction d’état lié (E<0) dans le puits fini
# ---------------------------------------------------------------------
def psi_bound(x, E_mod, V0, a):
        # κ = √(-E) pour décroissance exp. hors du puits
    κ = np.sqrt(-E_mod)
    q = np.sqrt(E_mod + V0)
    ψ_in  = np.cos(q * x)
    A_val = np.cos(q * a)
    ψ_out = A_val * np.exp(-κ * (np.abs(x) - a))
    
       # Assemblage et normalisation
    ψ     = np.where(np.abs(x) <= a, ψ_in, ψ_out)
    norm  = np.sqrt(np.trapezoid(np.abs(ψ)**2, x))
    return ψ / norm

# ---------------------------------------------------------------------
# 3) Tracé complet n=1..5 (états liés et continuum)
# ---------------------------------------------------------------------
xL, xR = -3*a_model, 3*a_model
x_grid = np.linspace(xL, xR, 2000)

# Potentiel en profil (affiché en pointillés gris)
V_x    = np.where(np.abs(x_grid) <= a_model, -V0_model, 0.0)

# Préparer la figure
fig, axes = plt.subplots(len(n_vals), 1, figsize=(8, 2.5*len(n_vals)), sharex=True)
for idx, n in enumerate(n_vals):
    Em = E_n_model[idx]
    ax = axes[idx]

    if Em < 0:
        # état lié
        ψ = psi_bound(x_grid, Em, V0_model, a_model)
        ax.plot(x_grid,      ψ,   'navy',    label="Re[ψ]")
        ax.plot(x_grid,    ψ**2,  'crimson', label="|ψ|²")
        ax.plot(x_grid, np.zeros_like(x_grid),
                'teal', linestyle='--', label="Im[ψ]")
    else:
          # état continuum (diffusion)
        r, B, C, D = solve_coefficients(Em, V0_model, a_model)
        Ψ = np.zeros_like(x_grid, dtype=complex)
        k = np.sqrt(Em)
        q = np.sqrt(Em + V0_model)

          # Reconstruction de Ψ point par point
        for j, xj in enumerate(x_grid):
            if xj < -a_model:
                Ψ[j] = np.exp(1j*k*xj) + r*np.exp(-1j*k*xj)
            elif xj > a_model:
                Ψ[j] = D*np.exp(1j*k*xj)
            else:
                Ψ[j] = B*np.exp(1j*q*xj) + C*np.exp(-1j*q*xj)
        ax.plot(x_grid, Ψ.real,    'navy',    label="Re[ψ]")
        ax.plot(x_grid, Ψ.imag,    'teal',    linestyle='--', label="Im[ψ]")
        ax.plot(x_grid, np.abs(Ψ)**2, 'crimson', label="|ψ|²")

 # potentiel
    ax.plot(x_grid, V_x/V0_model, 'gray', linestyle='--', label="V(x)/V0")

      # Titre et légende
    tag = "lié" if Em<0 else "continuum"
    Ej, Ee = E_n_real[idx], E_n_eV[idx]
    ax.set_title(f"n={n}, E={Ej:.3e} J ≃ {Ee:.3e} eV ({tag})")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("x (unités-modèle)")
plt.tight_layout()

# =====================================================================
# DEUXIÈME PARTIE : T(E), R(E) et pics T≈1 (Ramsauer–Townsend)
# =====================================================================
def compute_TR(E_vals_mod, V0, a):
    T_list = np.zeros_like(E_vals_mod)
    R_list = np.zeros_like(E_vals_mod)
    for i, Em in enumerate(E_vals_mod):
        r, _, _, D = solve_coefficients(Em, V0, a)
        T_list[i] = np.real(abs(D)**2) # coefficient de transmission
        R_list[i] = np.real(abs(r)**2) # coefficient de réflexion
    return T_list, R_list 

# Grille d’énergies pour tracer T(E) et R(E)
E_mod_vals = np.linspace(1e-3, 50.0, 2000)
T_vals, R_vals = compute_TR(E_mod_vals, V0_model, a_model)

# Conversion en unités réelles
E_real_vals   = E_mod_vals * E_unit
E_eV_vals     = E_real_vals / e_charge

# Repérer les pics où T≈1 (résonances)
eps = 1e-3
indices = np.where(T_vals > 1 - eps)[0]
blocks  = np.split(indices, np.where(np.diff(indices)!=1)[0]+1)

# Énergies des pics (max T dans chaque bloc)
E_peaks = [E_real_vals[blk][np.argmax(T_vals[blk])] for blk in blocks]
E_peaks_eV = [v/e_charge for v in E_peaks]

# Tracé de T(E) et R(E)
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.plot(E_real_vals, T_vals,   color='blue', label=r'$T(E)$')
ax2.plot(E_real_vals, R_vals, '--', color='red',  label=r'$R(E)$')
ax2.set_ylim(-0.05,1.05)
ax2.set_xlim(E_real_vals.min(), E_real_vals.max())
ax2.set_xlabel(r'Énergie réelle $E$ (J) — équivalent en eV')
ax2.set_ylabel(r'$T(E),\,R(E)$')
ax2.set_title('Effet Ramsauer–Townsend : $T(E)$ et $R(E)$')
ax2.grid(alpha=0.3)
ax2.legend(loc='upper right')

# Marquer les énergies analytiques (verticales oranges)
for Ej, Ee in zip(E_n_real, E_n_eV):
    ax2.axvline(Ej, color='orange', linestyle='--', alpha=0.6)
    ax2.text(Ej, 0.7, f"{Ej:.3e} J\n{Ee:.3e} eV", rotation=90,
             color='orange', fontsize=8, va='center')

plt.tight_layout()
plt.show()

