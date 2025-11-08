# fichier: ca_analysis.py
import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import fanalysis as CA

# ---------- Configuration ----------
INPUT_FILE = "Bilan Financier.xlsx"   
SHEET = 0                              # index de la feuille
ROW_VAR = "Product"
COL_VAR = "Segment"
N_COMPONENTS = 2                       # axe

# ---------- Charger données et construire tableau de contingence ----------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
# si nécessaire nettoyer les NA
df = df[[ROW_VAR, COL_VAR]].dropna()

cont = pd.crosstab(df[ROW_VAR], df[COL_VAR]).astype(float)
cont.index.name = ROW_VAR
cont.columns.name = COL_VAR

# Totaux
n = cont.values.sum()
P = cont / n                             # fréquences
r = P.sum(axis=1).values.reshape(-1,1)   # masses lignes (I x 1)
c = P.sum(axis=0).values.reshape(1,-1)   # masses colonnes (1 x J)

# Matrice des profils
# profils ligne p_ij / r_i 
profiles_line = P.div(r.flatten(), axis=0)   # DataFrame
profiles_col = P.div(c.flatten(), axis=1)    # DataFrame

# ---------- Construire la matrice des résidus standardisés S ----------
# centre = r @ c (outer product)
rc = r.dot(c)
S = (P - rc) / np.sqrt(r.dot(c))   # broadcast ok because shapes (I,1)*(1,J) -> (I,J)
# mais pour SVD il faut D_r^{-1/2} (P - rc) D_c^{-1/2}
Dr_inv_sqrt = np.diag((1.0 / np.sqrt(r.flatten())))
Dc_inv_sqrt = np.diag((1.0 / np.sqrt(c.flatten())))
M = Dr_inv_sqrt.dot((P - rc).values).dot(Dc_inv_sqrt)

# ---------- SVD ----------
U, sigma, VT = svd(M, full_matrices=False)
# sigma: singular values (length = min(I,J))
# valeurs propres (inerties) :
eigvals = sigma**2
# on garde N_COMPONENTS axes
K = min(N_COMPONENTS, len(sigma))

# Coordonnées (coord principales)
# lignes: F = D_r^{-1/2} U Sigma
F = Dr_inv_sqrt.dot(U[:, :K].dot(np.diag(sigma[:K])))    # shape (I, K)
# colonnes: G = D_c^{-1/2} V Sigma
V = VT.T
G = Dc_inv_sqrt.dot(V[:, :K].dot(np.diag(sigma[:K])))    # shape (J, K)

# DataFrames avec labels
row_coords = pd.DataFrame(F, index=cont.index, columns=[f"Dim{k+1}" for k in range(K)])
col_coords = pd.DataFrame(G, index=cont.columns, columns=[f"Dim{k+1}" for k in range(K)])

# ---------- Cos^2 (qualité de représentation) ----------
row_sqdist = (row_coords**2).sum(axis=1)   # somme des carrés des coords (distance² à l'origine)
col_sqdist = (col_coords**2).sum(axis=1)

row_cos2 = (row_coords**2).div(row_sqdist, axis=0).fillna(0)  # chaque ligne somme à 1 sur tous axes
col_cos2 = (col_coords**2).div(col_sqdist, axis=0).fillna(0)

# on peut extraire cos2 pour axes 1 et 2
row_cos2_axes = row_cos2.iloc[:, :K]
col_cos2_axes = col_cos2.iloc[:, :K]

# ---------- Contributions ----------
# contr_row_ik = (r_i * F_ik^2) / lambda_k
# lambda_k = eigvals[k]
eigvals_k = eigvals[:K]
# r_vector flatten
r_vec = r.flatten()
c_vec = c.flatten()

row_contrib = pd.DataFrame(index=cont.index, columns=[f"Dim{k+1}" for k in range(K)])
col_contrib = pd.DataFrame(index=cont.columns, columns=[f"Dim{k+1}" for k in range(K)])

for k in range(K):
    lam = eigvals_k[k]
    row_contrib.iloc[:, k] = (r_vec * (row_coords.iloc[:, k].values**2) / lam) * 100  # en pourcentage
    col_contrib.iloc[:, k] = (c_vec * (col_coords.iloc[:, k].values**2) / lam) * 100  # en %

# ---------- Résumé des valeurs propres et % inertie ----------
eigvals_full = eigvals
total_inertia = eigvals_full.sum()
perc_inertia = 100 * eigvals_full / total_inertia

eig_summary = pd.DataFrame({
    "eigvals": eigvals_full,
    "percent_inertia": perc_inertia,
    "cumulative_percent": 100 * np.cumsum(eigvals_full)/np.sum(eigvals_full)
})

# ---------- Sauvegarde résultats dans Excel ----------
with pd.ExcelWriter("results_CA3.xlsx") as writer:
    cont.to_excel(writer, sheet_name="contingency")
    P.to_excel(writer, sheet_name="frequencies")
    pd.DataFrame(r.flatten(), index=cont.index, columns=["mass_r"]).to_excel(writer, sheet_name="masses")
    row_coords.to_excel(writer, sheet_name="row_coords")
    col_coords.to_excel(writer, sheet_name="col_coords")
    row_cos2_axes.to_excel(writer, sheet_name="row_cos2_axes")
    col_cos2_axes.to_excel(writer, sheet_name="col_cos2_axes")
    row_contrib.to_excel(writer, sheet_name="row_contrib_%")
    col_contrib.to_excel(writer, sheet_name="col_contrib_%")
    eig_summary.to_excel(writer, sheet_name="eig_summary")

print("Fichier results_CA.xlsx créé avec tous les tableaux.") #message de confirmation d'aucune erreur 

# ---------- Graphiques ----------
# 1) Biplot (Dim1 x Dim2)
plt.figure(figsize=(9,7))
plt.axhline(0, color='grey', linewidth=0.8)
plt.axvline(0, color='grey', linewidth=0.8)
plt.scatter(row_coords["Dim1"], row_coords["Dim2"], c='blue', label='Rows (Segments)')
for i, txt in enumerate(row_coords.index):
    plt.annotate(txt, (row_coords["Dim1"].iat[i], row_coords["Dim2"].iat[i]), color='blue')
plt.scatter(col_coords["Dim1"], col_coords["Dim2"], c='red', marker='^', label='Columns (Countries)')
for j, txt in enumerate(col_coords.index):
    plt.annotate(txt, (col_coords["Dim1"].iat[j], col_coords["Dim2"].iat[j]), color='red')
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.title("CA biplot: Rows (blue) and Columns (red)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("CA_biplot.png", dpi=150)
plt.show()

# 2) Cos2 bar charts for rows on Dim1/Dim2
row_cos2_axes.plot(kind='bar', stacked=False, figsize=(10,6), title="Row cos2 (Dim1 & Dim2)")
plt.ylabel("cos^2")
plt.tight_layout()
plt.savefig("row_cos2.png", dpi=150)
plt.show()

# 3) Contributions (%) for rows on Dim1 and Dim2 (top contributors)
for k in range(K):
    s = row_contrib.iloc[:, k].sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    s.plot(kind='bar')
    plt.title(f"Row contributions (%) to Dim{k+1}")
    plt.ylabel("% contribution")
    plt.tight_layout()
    plt.savefig(f"row_contrib_dim{k+1}.png", dpi=150)
    plt.show()

# 4) Same for columns
col_cos2_axes.plot(kind='bar', stacked=False, figsize=(10,6), title="Column cos2 (Dim1 & Dim2)")
plt.ylabel("cos^2")
plt.tight_layout()
plt.savefig("col_cos2.png", dpi=150)
plt.show()

for k in range(K):
    s = col_contrib.iloc[:, k].sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    s.plot(kind='bar')
    plt.title(f"Column contributions (%) to Dim{k+1}")
    plt.ylabel("% contribution")
    plt.tight_layout()
    plt.savefig(f"col_contrib_dim{k+1}.png", dpi=150)
    plt.show()

print("Graphiques sauvegardés: CA_biplot.png, row_cos2.png, col_cos2.png, row_contrib_dim*.png, col_contrib_dim*.png") #message pour dire a fait part de la disponibilite des images a l'user 

