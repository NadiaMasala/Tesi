import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. Generazione di un dataset non etichettato con 3 cluster
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=82)


# 2. Partizionamento tramite K-Means (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)
centers_kmeans = kmeans.cluster_centers_

# 3. Partizionamento tramite DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=3)
labels_dbscan = dbscan.fit_predict(X)

# 4. Visualizzazione e confronto dei risultati
fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=300)

# Punti
axes[0].scatter(X[:, 0], X[:, 1], s=40, edgecolors='k')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Grafico k-Means
axes[1].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=40, edgecolors='k')
axes[1].scatter(centers_kmeans[:, 0], centers_kmeans[:, 1], c='red', marker='X', s=200, label='Centroids')
axes[1].set_title("k-Means (k=3)")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

# Grafico DBSCAN
scatter_db = axes[2].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', s=40, edgecolors='k')
axes[2].set_title("DBSCAN")
axes[2].grid(True, linestyle='--', alpha=0.5)

# Evidenziazione di eventuale rumore identificato da DBSCAN (label = -1)
n_noise = np.sum(labels_dbscan == -1)
if n_noise > 0:
    axes[2].text(0.97, 0.07, f'Noise points: {n_noise}', transform=axes[2].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


plt.savefig('images/k-means_dbscan.pdf')

quit()
'''
import matplotlib.pyplot as plt
import numpy as np

# Configurazione della figura e pulizia degli assi
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.set_aspect('equal')
ax.axis('off')

# 1. Cluster di punti
blue_cluster = np.array([
    [2.0, 6.4], [2.2, 7.2], [2.7, 7.2], [2.8, 6.7],
    [1.7, 5.7], [2.3, 5.9], [2.9, 6.3], [2.2, 5.2],
    [2.7, 5.3], [3.0, 6.8]
])

pink_cluster = np.array([
    [4.9, 5.5], [5.3, 6.7], [5.8, 6.3], [5.8, 5.2],
    [6.4, 6.3], [6.6, 5.8], [6.8, 4.9], [6.2, 4.9],
    [5.8, 4.6], [5.5, 3.9]
])

green_cluster = np.array([
    [2.8, 3.0], [2.5, 2.3], [3.3, 2.6], [3.2, 2.0],
    [3.7, 2.1], [3.9, 2.8], [4.2, 1.9], [4.6, 2.4],
    [4.5, 1.2], [5.3, 1.4], [5.2, 2.3]
])

# 2. Parametri delle rette di separazione: y = m*x + q
# Per OvR usiamo la forma canonica w1*x + w2*y + b > 0 verso la classe target
m_b, q_b = 1.35, 0.95
m_p, q_p = -2.7, 17.0
m_g, q_g = -0.22, 4.3

# 3. Creazione della griglia per la colorazione continua
x_min, x_max = 1.0, 7.3
y_min, y_max = 0.5, 8.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 600),
                     np.linspace(y_min, y_max, 600))

# Punteggi per la regola argmax (valore positivo nell'area della rispettiva classe)
score_blue  =  yy - (m_b * xx + q_b)
score_pink  =  yy - (m_p * xx + q_p)
score_green = -(yy - (m_g * xx + q_g))

# Assegnazione di ogni pixel della griglia al punteggio massimo
scores = np.stack([score_blue, score_pink, score_green], axis=-1)
regions = np.argmax(scores, axis=-1)

# 4. Colorazione dello sfondo senza regioni ambigue
colors_bg = ['#c2daff', '#ffc2d4', '#c2ffd9'] # Blu, Rosa e Verde tenui

ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=colors_bg, alpha=0.95)

# 5. Tracciamento delle tre rette con doppio bordo
x_vals = np.linspace(x_min, x_max, 200)
y_blue_line = m_b * x_vals + q_b
y_pink_line = m_p * x_vals + q_p
y_green_line = m_g * x_vals + q_g

def plot_bordered_line(x_v, y_v, line_color):
    ax.plot(x_v, y_v, color='#1a1a1a', linewidth=4.5, zorder=2)
    ax.plot(x_v, y_v, color=line_color, linewidth=2.2, zorder=3)

plot_bordered_line(x_vals, y_blue_line, '#0066ff')
plot_bordered_line(x_vals, y_pink_line, '#ff1a66')
plot_bordered_line(x_vals, y_green_line, '#00ff99')

# 6. Scatter plot dei punti
point_size = 50
linewidth_edge = 1.5

ax.scatter(blue_cluster[:, 0], blue_cluster[:, 1], color='#0066ff', edgecolors='#1a1a1a',
           s=point_size, linewidths=linewidth_edge, zorder=4)

ax.scatter(pink_cluster[:, 0], pink_cluster[:, 1], color='#ff1a66', edgecolors='#1a1a1a',
           s=point_size, linewidths=linewidth_edge, zorder=4)

ax.scatter(green_cluster[:, 0], green_cluster[:, 1], color='#00ff99', edgecolors='#1a1a1a',
           s=point_size, linewidths=linewidth_edge, zorder=4)

# 7. Limiti della figura
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('images/svm_multi_fig2.pdf')


import matplotlib.pyplot as plt
import numpy as np

# Configurazione della figura e pulizia degli assi
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.set_aspect('equal')
ax.axis('off')

# 1. Cluster di punti ricalibrati
# Cluster Blu (in alto a sinistra)
blue_cluster = np.array([
    [2.0, 6.4], [2.2, 7.2], [2.7, 7.2], [2.8, 6.7],
    [1.7, 5.7], [2.3, 5.9], [2.9, 6.3], [2.2, 5.2],
    [2.7, 5.3], [3.0, 6.8]
])

# Cluster Rosa/Rosso (in alto a destra)
pink_cluster = np.array([
    [4.9, 5.5], [5.3, 6.7], [5.8, 6.3], [5.8, 5.2],
    [6.4, 6.3], [6.6, 5.8], [6.8, 4.9], [6.2, 4.9],
    [5.8, 4.6], [5.5, 3.9]
])

# Cluster Verde Menta (in basso)
green_cluster = np.array([
    [2.8, 3.0], [2.5, 2.3], [3.3, 2.6], [3.2, 2.0],
    [3.7, 2.1], [3.9, 2.8], [4.2, 1.9], [4.6, 2.4],
    [4.5, 1.2], [5.3, 1.4], [5.2, 2.3]
])

# 2. Tracciamento dei punti con bordo scuro marcato
point_size = 50
linewidth_edge = 1.5

ax.scatter(blue_cluster[:, 0], blue_cluster[:, 1], color='#0066ff', edgecolors='#1a1a1a',
           s=point_size, linewidths=linewidth_edge, zorder=3)

ax.scatter(pink_cluster[:, 0], pink_cluster[:, 1], color='#ff1a66', edgecolors='#1a1a1a',
           s=point_size, linewidths=linewidth_edge, zorder=3)

ax.scatter(green_cluster[:, 0], green_cluster[:, 1], color='#00ff99', edgecolors='#1a1a1a',
           s=point_size, linewidths=linewidth_edge, zorder=3)

# 3. Rette di separazione SVM One-vs-Rest
x = np.linspace(0, 8, 200)

# Retta Blu: traslata più in basso (intercetta q = 0.95 invece di -0.1)
# per riflettere il margine equidistante tra i punti blu e il blocco rosa+verde
y_blue = 1.35 * x + 0.95

# Retta Rosa: separa il cluster rosa dal resto
y_pink = -2.7 * x + 17.0

# Retta Verde: separa il cluster verde dal resto
y_green = -0.22 * x + 4.3

# Funzione per tracciare le rette con bordo scuro spesso
def plot_bordered_line(x_vals, y_vals, line_color):
    ax.plot(x_vals, y_vals, color='#1a1a1a', linewidth=5, zorder=1)      # Bordo nero
    ax.plot(x_vals, y_vals, color=line_color, linewidth=2.5, zorder=2) # Anima colorata

plot_bordered_line(x, y_blue, '#0066ff')
plot_bordered_line(x, y_pink, '#ff1a66')
plot_bordered_line(x, y_green, '#00ff99')

# 4. Limiti del grafico
ax.set_xlim(1.0, 7.3)
ax.set_ylim(0.5, 8.5)

plt.tight_layout()
plt.savefig('images/svm_multi_fig1.pdf')
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 1. Definizione dei punti dati per far toccare esattamente i margini (Z = ±1)
X_0 = np.array([
    [-1.5, 2.1], [-0.5, 1.2], [-1.1, -0.2], [-1.2, -0.4],
    [-1.4, -0.9], [-1.5, -1.0], [-1.3, -1.2],
    [0.38, -0.71]   # Support Vector viola: cade esattamente sulla linea tratteggiata (Z = -1)
])

X_1 = np.array([
    [1.3, 2.1],
    [0.98, 1.02],   # Support Vector giallo in alto: cade esattamente sulla linea tratteggiata (Z = +1)
    [1.18, 0.52],   # Support Vector giallo centrale: cade esattamente sulla linea tratteggiata (Z = +1)
    [1.3, 0.8],
    [0.18, -2.02],  # Support Vector giallo in basso: cade esattamente sulla linea tratteggiata (Z = +1)
    [0.2, -2.3], [0.0, -2.7], [0.5, -2.4]
])

X = np.vstack((X_0, X_1))
y = np.array([0]*len(X_0) + [1]*len(X_1))

# 2. Addestramento SVM con Kernel Lineare (Hard Margin / C elevato)
clf = SVC(kernel='linear')
clf.fit(X, y)

# 3. Creazione della griglia di valutazione
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

x_min, x_max = -3, 3
y_min, y_max = -3, 3
fill_x_min, fill_x_max = -2.5, 2.3

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_dec = clf.decision_function(grid_points).reshape(xx.shape)
Z_pred = clf.predict(grid_points).reshape(xx.shape)

# Maschera per limitare lo sfondo colorato alla banda [-2.5, 2.3]
mask = (xx >= fill_x_min) & (xx <= fill_x_max)
Z_pred_masked = np.where(mask, Z_pred, np.nan)

# 4. Colorazione distinta delle regioni di decisione (Viola vs Giallo)
ax.contourf(xx, yy, Z_pred_masked, levels=[-0.5, 0.5, 1.5],
            colors=['#c2b2cd', '#fffab3'], alpha=0.95)

# 5. Confine decisionale rettilineo (Z=0) e iperpiani di margine (Z=-1, Z=1 tratteggiati)
ax.contour(xx, yy, Z_dec, levels=[-1.0, 0.0, 1.0],
           colors='k', linestyles=['--', '-', '--'], linewidths=[1.5, 1.8, 1.5], zorder=3)

# 6. Evidenziazione dei Vettori di Supporto con cerchio centrato
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=250, facecolors='none', edgecolors='k', linewidths=1.2, zorder=4)

# 7. Disegno dei punti delle due classi
ax.scatter(X_0[:, 0], X_0[:, 1], c='#440154', edgecolors='k', s=50, zorder=5, label='0')
ax.scatter(X_1[:, 0], X_1[:, 1], c='#fde725', edgecolors='k', s=50, zorder=5, label='1')

# 8. Limiti degli assi e legenda
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend(title="Classes", loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('images/svm_lin_fig.pdf')


quit()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 1. Definizione dei punti dati per far toccare esattamente i margini (Z = ±1)
X_0 = np.array([
    [-1.5, 2.1], [-0.5, 1.2], [-1.1, -0.2], [-1.2, -0.4],
    [-1.4, -0.9], [-1.5, -1.0], [-1.3, -1.2],
    [0.38, -0.71]   # Support Vector viola: cade esattamente sulla linea tratteggiata (Z = -1)
])

X_1 = np.array([
    [1.3, 2.1],
    [0.98, 1.02],   # Support Vector giallo in alto: cade esattamente sulla linea tratteggiata (Z = +1)
    [1.18, 0.52],   # Support Vector giallo centrale: cade esattamente sulla linea tratteggiata (Z = +1)
    [1.3, 0.8],
    [0.18, -2.02],  # Support Vector giallo in basso: cade esattamente sulla linea tratteggiata (Z = +1)
    [0.2, -2.3], [0.0, -2.7], [0.5, -2.4]
])

X = np.vstack((X_0, X_1))
y = np.array([0]*len(X_0) + [1]*len(X_1))

# 2. Addestramento SVM Polinomiale (Grado 3)
clf = SVC(kernel='poly', degree=3, C=10.0, coef0=1.0, gamma='scale')
clf.fit(X, y)

# 3. Creazione della griglia di valutazione
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

x_min, x_max = -3, 3
y_min, y_max = -3, 3
fill_x_min, fill_x_max = -2.5, 2.3

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_dec = clf.decision_function(grid_points).reshape(xx.shape)
Z_pred = clf.predict(grid_points).reshape(xx.shape)

# Maschera per limitare lo sfondo colorato alla banda [-2.5, 2.3]
mask = (xx >= fill_x_min) & (xx <= fill_x_max)
Z_pred_masked = np.where(mask, Z_pred, np.nan)

# 4. Colorazione distinta delle regioni di decisione (Viola vs Giallo)
ax.contourf(xx, yy, Z_pred_masked, levels=[-0.5, 0.5, 1.5],
            colors=['#c2b2cd', '#fffab3'], alpha=0.95)

# 5. Confine decisionale (Z=0 continua) e margini (Z=-1, Z=1 tratteggiati)
ax.contour(xx, yy, Z_dec, levels=[-1.0, 0.0, 1.0],
           colors='k', linestyles=['--', '-', '--'], linewidths=[1.5, 1.8, 1.5], zorder=3)

# 6. Evidenziazione dei Vettori di Supporto con cerchio centrato
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=250, facecolors='none', edgecolors='k', linewidths=1.2, zorder=4)

# 7. Disegno dei punti delle due classi
ax.scatter(X_0[:, 0], X_0[:, 1], c='#440154', edgecolors='k', s=50, zorder=5, label='0')
ax.scatter(X_1[:, 0], X_1[:, 1], c='#fde725', edgecolors='k', s=50, zorder=5, label='1')

# 8. Limiti degli assi e legenda
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend(title="Classes", loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('images/svm_ker3_fig.pdf')

quit()

import matplotlib.pyplot as plt
import numpy as np

# Configurazione della figura e pulizia degli assi
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.set_aspect('equal')
ax.axis('off')

# Coordinate dei punti blu (classe superiore/destra)
blue_points = np.array([
    [3.9, 8.0],
    [5.0, 7.4],
    [4.4, 6.3],
    [5.3, 6.3],
    [6.4, 6.8],
    [5.0, 4.6],
    [6.4, 5.0],
    [5.8, 3.8]
])

# Coordinate dei punti rossi (classe inferiore/sinistra)
red_points = np.array([
    [1.5, 5.0],
    [2.4, 5.7],
    [3.0, 4.3],
    [2.3, 3.2],
    [1.5, 2.4],
    [3.3, 2.8],
    [4.4, 1.6]
])

# Parametri della retta separatrice: y = m*x + q
x_line = np.linspace(1.1, 6.1, 100)
y_line = -0.8 * x_line + 8.1

# Tracciamento dei punti con bordo scuro
ax.scatter(red_points[:, 0], red_points[:, 1], color='#a62626', s=50)
ax.scatter(blue_points[:, 0], blue_points[:, 1], color='#2b3a7a', s=50)

# Tracciamento della retta (si interrompe prima del testo per non sovrapporsi)
ax.plot(x_line, y_line, color='#1a1a1a', linewidth=2, zorder=2)

# Aggiunta dell'equazione esattamente al termine della retta
ax.text(6.2, 2.6, r'w·x + b=0', fontsize=20,
        verticalalignment='center', horizontalalignment='left', color='#1a1a1a')

# Limiti del grafico
ax.set_xlim(0.5, 9.5)
ax.set_ylim(0.5, 8.8)

plt.tight_layout()
plt.savefig('images/svm_fig_1.pdf')

quit()

import matplotlib.pyplot as plt
import numpy as np

# Configurazione della figura e pulizia degli assi
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.set_aspect('equal')
ax.axis('off')

# Coordinate dei punti blu (classe superiore/destra)
blue_points = np.array([
    [3.9, 8.0],
    [5.0, 7.4],
    [4.4, 6.3],
    [5.3, 6.3],
    [6.4, 6.8],
    [5.05, 4.6],  # Support vector blu
    [6.4, 5.0],
    [5.8, 3.8]
])

# Coordinate dei punti rossi (classe inferiore/sinistra)
red_points = np.array([
    [1.5, 5.0],
    [2.7, 5.7],  # Support vector rosso
    [3.0, 4.3],
    [2.3, 3.2],
    [1.5, 2.4],
    [3.3, 2.8],
    [4.4, 1.6]
])

# --- GEOMETRIA RETTE CON INCLINAZIONE RIPIDA ---
m = -1.35
# Distanza verticale calibrata sul raggio dei punti per farli sfiorare perfettamente
offset = 0.88
q_center = 10.38

# 1. Tracciamento delle tre rette
# Retta centrale (w·x + b = 0)
x_center = np.linspace(2.5, 5.7, 100)
y_center = m * x_center + q_center
ax.plot(x_center, y_center, color='#1a1a1a', linewidth=2, zorder=2)

# Retta tratteggiata superiore (w·x + b = +1)
x_upper = np.linspace(2.8, 5.8, 100)
y_upper = m * x_upper + (q_center + offset)
ax.plot(x_upper, y_upper, color='#1a1a1a', linestyle='--', linewidth=1.5, zorder=2)

# Retta tratteggiata inferiore (w·x + b = -1)
x_lower = np.linspace(2.2, 5.2, 100)
y_lower = m * x_lower + (q_center - offset)
ax.plot(x_lower, y_lower, color='#1a1a1a', linestyle='--', linewidth=1.5, zorder=2)

# 2. Plot dei punti con bordo scuro
ax.scatter(red_points[:, 0], red_points[:, 1], color='#a62626', s=50)
ax.scatter(blue_points[:, 0], blue_points[:, 1], color='#2b3a7a', s=50)

# 3. Freccia del margine (perpendicolare alla retta centrale)
x_start = 5.05
y_start = m * x_start + q_center
# Vettore ortogonale (pendenza = -1/m)
dx = offset / (-m - (1/m))
dy = -1/m * dx
x_end = x_start + dx
y_end = y_start + dy

ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
            arrowprops=dict(arrowstyle='<->', color='#a62626', lw=1.8, mutation_scale=12), zorder=5)

# 4. Etichette ed equazioni traslate all'esterno per evitare sovrapposizioni
# Equazione centrale in alto a sinistra
ax.text(-0.1, 7.3, r'w·x + b = 0', fontsize=18, color='#1a1a1a', verticalalignment='center')

# Equazione tratteggiata superiore
ax.text(5.9, 3.2, r'w·x + b = +1', fontsize=18, color='#1a1a1a', verticalalignment='center')

# Equazione tratteggiata inferiore
ax.text(2.2, 2.1, r'w·x + b = -1', fontsize=18, color='#1a1a1a', verticalalignment='center')

# Scritta "margin"
ax.text(5.6, 4.2, 'margin', fontsize=16, color='#a62626', fontweight='bold', verticalalignment='center')

# Limiti della figura
ax.set_xlim(0.5, 9.5)
ax.set_ylim(0.5, 9.2)

plt.tight_layout()
plt.savefig('images/svm_fig_2.pdf')
