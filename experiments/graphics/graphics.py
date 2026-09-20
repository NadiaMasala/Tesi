import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from Spherical_Class_class import Spherical_Classifier
from sklearn.svm import SVC

# 1. Dati degli esperimenti
exp = [
 "150_2_3",
 "150_2_5",
 "150_2_7",
 "150_5_3",
 "150_5_5",
 "150_5_7",
 "200_2_3",
 "200_2_5",
 "200_2_7",
 "200_5_3",
 "200_5_5",
 "200_5_7",
 "300_2_3",
 "300_2_5",
 "300_5_3",
 "300_5_5"
]

DBSC_sc = [0.784, 0.754, 0.702, 0.804, 0.635, 0.596, 0.783, 0.755, 0.692, 0.822, 0.722, 0.668, 0.779, 0.746, 0.805, 0.672]
DBSC_dbscan = [0.99, 0.949, 0.974, 0.952, 0.937, 0.866, 0.978, 0.971, 0.932, 0.956, 0.949, 0.977, 0.888, 0.993, 0.972, 0.929]

# 2. Posizionamento delle barre sull'asse X
x = np.arange(len(exp))  # Indici per gli esperimenti
larghezza = 0.35  # Larghezza di ciascuna barra

fig, ax = plt.subplots(figsize=(9, 6))

# 3. Creazione delle barre affiancate
barre_DBSC_sc = ax.bar(
 x - larghezza / 2,
 DBSC_sc,
 larghezza,
 label="Clustering Sferico",
 color="#2b5c8f",
)
barre_DBSC_dbscan = ax.bar(
 x + larghezza / 2, DBSC_dbscan, larghezza, label="DBSCAN", color="#d95f02"
)

# 4. Personalizzazione del grafico
ax.set_xticks(x)
ax.set_xticklabels(exp, rotation=45, ha="right", fontsize=10)
ax.set_ylim(0, 1.05)  # Scala da 0 a 1 (con un po' di margine in alto)
ax.set_ylabel('DBSC_avg')
ax.legend(loc="lower left",fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)


# # # 5. Aggiunta dei valori sopra ad ogni barra (opzionale)
# # def aggiungi_valori(barre):
# #     for barra in barre:
# #         altezza = barra.get_height()
# #         ax.annotate(
# #             f"{altezza:.3f}",
# #             xy=(barra.get_x() + barra.get_width() / 2, altezza),
# #             xytext=(0, 3),  # Offset verticale
# #             textcoords="offset points",
# #             ha="center",
# #             va="bottom",
# #             fontsize=9,
# #         )
# #
# # aggiungi_valori(barre_acc_sc)
# # aggiungi_valori(barre_acc_ksvm)
#
plt.tight_layout()
plt.savefig('new_DBSC_sc_dbscan.pdf')


quit()

# # 1. Dati degli esperimenti
# exp = [
#     "australian",
#     "blood_transfusion",
#     "breast",
#     "breast_wisconsin",
#     "diabetes",
#     "divorce",
#     "flowmeters",
#     "Gallstone",
#     "liver",
#     "Mesothelioma",
#     "sonar"
# ]
#
# accuracy_sc = [0.732, 0.7, 0.92, 0.877, 0.714, 1.0, 0.444, 0.578, 0.621, 0.708, 0.524]
# f1_score_sc = [0.755, 0.118, 0.871, 0.91, 0.796, 1.0, 0.167, 0.597, 0.154, 0.829, 0.63]
#
# accuracy_ksvm = [0.891, 0.767, 0.993, 0.974, 0.799, 1.0, 0.778, 0.828, 0.621, 1.0, 0.905]
# f1_score_ksvm = [0.882, 0.103, 0.99, 0.979, 0.858, 1.0, 0.6, 0.831, 0.353, 1.0, 0.9]
#
# # 2. Posizionamento delle barre sull'asse X
# x = np.arange(len(exp))  # Indici per gli esperimenti
# larghezza = 0.35  # Larghezza di ciascuna barra
#
# fig, ax = plt.subplots(figsize=(9, 6))
#
# # 3. Creazione delle barre affiancate
# barre_acc_sc = ax.bar(
#     x - larghezza / 2,
#     accuracy_sc,
#     larghezza,
#     label="Classificatore Sferico",
#     color="#2b5c8f",
# )
# barre_acc_ksvm = ax.bar(
#     x + larghezza / 2, accuracy_ksvm, larghezza, label="Kernel SVM", color="#d95f02"
# )
#
# # 4. Personalizzazione del grafico
# ax.set_xticks(x)
# ax.set_xticklabels(exp, rotation=45, ha="right", fontsize=10)
# ax.set_ylim(0, 1.05)  # Scala da 0 a 1 (con un po' di margine in alto)
# ax.set_ylabel('Accuratezza')
# ax.legend(loc="best",fontsize=11)
# ax.grid(axis="y", linestyle="--", alpha=0.5)
#
#
# # # 5. Aggiunta dei valori sopra ad ogni barra (opzionale)
# # def aggiungi_valori(barre):
# #     for barra in barre:
# #         altezza = barra.get_height()
# #         ax.annotate(
# #             f"{altezza:.3f}",
# #             xy=(barra.get_x() + barra.get_width() / 2, altezza),
# #             xytext=(0, 3),  # Offset verticale
# #             textcoords="offset points",
# #             ha="center",
# #             va="bottom",
# #             fontsize=9,
# #         )
# #
# # aggiungi_valori(barre_acc_sc)
# # aggiungi_valori(barre_acc_ksvm)
#
# plt.tight_layout()
# plt.savefig('acc_sc_ksvm.pdf')
#
# fig, ax = plt.subplots(figsize=(9, 6))
#
# # 3. Creazione delle barre affiancate
# barre_f1_sc = ax.bar(
#     x - larghezza / 2,
#     f1_score_sc,
#     larghezza,
#     label="Classificatore Sferico",
#     color="#2b5c8f",
# )
# barre_f1_ksvm = ax.bar(
#     x + larghezza / 2, f1_score_ksvm, larghezza, label="Kernel SVM", color="#d95f02"
# )
#
# # 4. Personalizzazione del grafico
# ax.set_xticks(x)
# ax.set_xticklabels(exp, rotation=45, ha="right", fontsize=10)
# ax.set_ylim(0, 1.05)  # Scala da 0 a 1 (con un po' di margine in alto)
# ax.set_ylabel('F1-score')
# ax.legend(loc="best",fontsize=11)
# ax.grid(axis="y", linestyle="--", alpha=0.5)
#
# # aggiungi_valori(barre_f1_sc)
# # aggiungi_valori(barre_f1_ksvm)
#
# plt.tight_layout()
# plt.savefig('f1_sc_ksvm.pdf')
#
#
# quit()

# 1. Dati degli esperimenti
exp_mb = [
    "100_2",
    "100_10",
    "100_40",
    "200_2",
    "200_10",
    "200_40"
]

accuracy_mb_train = [0.988, 0.9, 1.0, 0.9, 1.0, 0.838]
f1_score_mb_train = [0.987, 0.889, 1.0, 0.901, 1.0, 0.86]
accuracy_mb = [1.0, 0.9, 0.95, 0.8, 0.975, 0.825]
f1_score_mb = [1.0, 0.889, 0.952, 0.8, 0.976, 0.851]

# 2. Configurazione delle posizioni e della larghezza
x = np.arange(len(exp_mb))
n_barre = 4
larghezza = 0.18  # Ridotta la larghezza per far entrare 4 colonne
gap = 0.08

fig, ax = plt.subplots(figsize=(10, 6))

# 3. Posizionamento delle 4 barre con i rispettivi offset
# I 4 offset bilanciati attorno al centro x sono: -1.5, -0.5, +0.5, +1.5
ax.bar(
    x - 1.5 * larghezza - gap / 2,
    accuracy_mb_train,
    larghezza,
    label="Accuratezza training",
    color="#2b5c8f",
)
ax.bar(
    x - 0.5 * larghezza - gap / 2,
    accuracy_mb,
    larghezza,
    label="Accuratezza test",
    color="#41b6c4",
)
ax.bar(
    x + 0.5 * larghezza + gap / 2,
    f1_score_mb_train,
    larghezza,
    label="F1-score training",
    color="#fecc5c",
)
ax.bar(
    x + 1.5 * larghezza + gap / 2,
    f1_score_mb,
    larghezza,
    label="F1-score test",
    color="#d95f02",
)

# 4. Personalizzazione
ax.set_xticks(x)
ax.set_xticklabels(exp_mb, fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc="lower left")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig('acc_f1_mb_tr_ts.pdf')

quit()

# 1. Dati degli esperimenti
exp_mb = [
    "100_2",
    "100_10",
    "100_40",
    "200_2",
    "200_10",
    "200_40"
]

accuracy_mb = [1.0, 0.9, 0.95, 0.8, 0.975, 0.825]
f1_score_mb = [1.0, 0.889, 0.952, 0.8, 0.976, 0.851]

# 2. Posizionamento delle barre sull'asse X
x = np.arange(len(exp_mb))  # Indici per gli esperimenti
larghezza = 0.35  # Larghezza di ciascuna barra

fig, ax = plt.subplots(figsize=(9, 6))

# 3. Creazione delle barre affiancate
barre_acc = ax.bar(
    x - larghezza / 2,
    accuracy_mb,
    larghezza,
    label="Accuratezza",
    color="#2b5c8f",
)
barre_f1 = ax.bar(
    x + larghezza / 2, f1_score_mb, larghezza, label="F1-score", color="#d95f02"
)

# 4. Personalizzazione del grafico
ax.set_xticks(x)
ax.set_xticklabels(exp_mb, fontsize=10)
ax.set_ylim(0, 1.05)  # Scala da 0 a 1 (con un po' di margine in alto)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)


# 5. Aggiunta dei valori sopra ad ogni barra (opzionale)
def aggiungi_valori(barre):
    for barra in barre:
        altezza = barra.get_height()
        ax.annotate(
            f"{altezza:.3f}",
            xy=(barra.get_x() + barra.get_width() / 2, altezza),
            xytext=(0, 3),  # Offset verticale
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


aggiungi_valori(barre_acc)
aggiungi_valori(barre_f1)

plt.tight_layout()
plt.savefig('acc_f1_mb.pdf')

# 1. Dati degli esperimenti
exp_mb = [
    "100_2",
    "100_10",
    "100_40",
    "200_2",
    "200_10",
    "200_40"
]

accuracy_mb_train = [0.988, 0.9, 1.0, 0.9, 1.0, 0.838]
f1_score_mb_train = [0.987, 0.889, 1.0, 0.901, 1.0, 0.86]
accuracy_mb = [1.0, 0.9, 0.95, 0.8, 0.975, 0.825]
f1_score_mb = [1.0, 0.889, 0.952, 0.8, 0.976, 0.851]

# 2. Configurazione delle posizioni e della larghezza
x = np.arange(len(exp_mb))
n_barre = 4
larghezza = 0.18  # Ridotta la larghezza per far entrare 4 colonne

fig, ax = plt.subplots(figsize=(10, 6))

# 3. Posizionamento delle 4 barre con i rispettivi offset
# I 4 offset bilanciati attorno al centro x sono: -1.5, -0.5, +0.5, +1.5
ax.bar(
    x - 1.5 * larghezza,
    accuracy_mb_train,
    larghezza,
    label="Accuratezza training",
    color="#2b5c8f",
)
ax.bar(
    x - 0.5 * larghezza,
    accuracy_mb,
    larghezza,
    label="Accuratezza test",
    color="#41b6c4",
)
ax.bar(
    x + 0.5 * larghezza,
    f1_score_mb_train,
    larghezza,
    label="F1-score training",
    color="#fecc5c",
)
ax.bar(
    x + 1.5 * larghezza,
    f1_score_mb,
    larghezza,
    label="F1-score test",
    color="#d95f02",
)

# 4. Personalizzazione
ax.set_xticks(x)
ax.set_xticklabels(exp_mb, fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc="lower left")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig('acc_f1_mb_tr_ts.pdf')

quit()

R=2
#plt.use('TkAgg')  # rendo il grafico interattivo
figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
# Parametrizzazione della sfera
theta = np.linspace(0, 2 * np.pi, 20)
phi = np.linspace(0, np.pi, 20)
x = R * np.outer(np.sin(phi), np.cos(theta))
y = R * np.outer(np.sin(phi), np.sin(theta))
z = R * np.outer(np.cos(phi), np.ones_like(theta))
# Grafico 3D della sfera ottimale
axes.plot_surface(x, y, z, color='white', edgecolor='grey', alpha=0.3)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('z')

# 4. Piano di equazione x = 1 (ortogonale all'asse X)
y_plane = np.linspace(-2.2, 2.2, 10)
z_plane = np.linspace(-2.2, 2.2, 10)
Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
X_plane = np.ones_like(Y_plane)  # x costante pari a 1

axes.plot_surface(X_plane, Y_plane, Z_plane, color="lightskyblue", alpha=0.25)

# 5. Circonferenza di intersezione vuota (solo contorno rosso)
r_circle = np.sqrt(R**2 - 1**2)  # raggio = sqrt(3)
the = np.linspace(0, 2 * np.pi, 100)
x_circle = np.ones_like(the)  # x = 1
y_circle = r_circle * np.cos(the)
z_circle = r_circle * np.sin(the)

axes.plot(x_circle, y_circle, z_circle, color="red", linewidth=3)

# 6. Etichette e impostazione dei Ticks sugli assi
axes.set_xlabel("x", fontsize=12, fontweight="bold")
axes.set_ylabel("y", fontsize=12, fontweight="bold")
axes.set_zlabel("z", fontsize=12, fontweight="bold")

ticks = [-2, -1, 0, 1, 2]
axes.set_xticks(ticks)
axes.set_yticks(ticks)
axes.set_zticks(ticks)

plt.savefig('sfera_piano.pdf')

quit()
# 1. Inizializzazione della figura 3D con sfondo bianco
fig = plt.figure(figsize=(9, 9), facecolor="white")
ax = fig.add_subplot(111, projection="3d", facecolor="white")

# Disabilita le griglie trasparenti del box 3D di Matplotlib
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("white")
ax.yaxis.pane.set_edgecolor("white")
ax.zaxis.pane.set_edgecolor("white")

# 2. Assi cartesiani passanti per l'origine (0, 0, 0)
lim = 2.5
ax.plot([-lim, lim], [0, 0], [0, 0], color="black", linewidth=1.2)  # Asse X
ax.plot([0, 0], [-lim, lim], [0, 0], color="black", linewidth=1.2)  # Asse Y
ax.plot([0, 0], [0, 0], [-lim, lim], color="black", linewidth=1.2)  # Asse Z

# 3. Sfera di colore chiaro e molto trasparente (R = 2)
R = 2
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)
x_sphere = R * np.outer(np.cos(u), np.sin(v))
y_sphere = R * np.outer(np.sin(u), np.sin(v))
z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(
    x_sphere,
    y_sphere,
    z_sphere,
    color="lightskyblue",
    alpha=0.1,
    rstride=2,
    cstride=2,
    edgecolor="none",
)

# 4. Piano di equazione x = 1 (ortogonale all'asse X)
y_plane = np.linspace(-2.2, 2.2, 10)
z_plane = np.linspace(-2.2, 2.2, 10)
Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
X_plane = np.ones_like(Y_plane)  # x costante pari a 1

ax.plot_surface(X_plane, Y_plane, Z_plane, color="lightblue", alpha=0.25)

# 5. Circonferenza di intersezione vuota (solo contorno rosso)
r_circle = np.sqrt(R**2 - 1**2)  # raggio = sqrt(3)
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = np.ones_like(theta)  # x = 1
y_circle = r_circle * np.cos(theta)
z_circle = r_circle * np.sin(theta)

ax.plot(x_circle, y_circle, z_circle, color="red", linewidth=3)

# 6. Etichette e impostazione dei Ticks sugli assi
ax.set_xlabel("x", fontsize=12, fontweight="bold")
ax.set_ylabel("y", fontsize=12, fontweight="bold")
ax.set_zlabel("z", fontsize=12, fontweight="bold")

ticks = [-2, -1, 0, 1, 2]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])

# Inclinazione della vista per osservare bene il piano x = 1
ax.view_init(elev=20, azim=30)

plt.show()


quit()

exp_mb = [
    "100_2",
    "100_10",
    "100_40",
    "200_2",
    "200_10",
    "200_40"
]

accuracy_mb = [1.0, 0.9, 0.95, 0.8, 0.975, 0.825]
f1_score_mb = [1.0, 0.889, 0.952, 0.8, 0.976, 0.851]

y = np.arange(len(exp_mb))

bar_height = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

blue = "#5B9BD5"
red = "#E57373"

# Accuracy sopra
ax.barh(
    y - bar_height/2,
    accuracy_mb,
    height=bar_height,
    label="accuracy",
    color=red
)

# F1 Score sotto
ax.barh(
    y + bar_height/2,
    f1_score_mb,
    height=bar_height,
    label="F1",
    color=blue
)

ax.set_yticks(y)
for i, exp in enumerate(exp_mb):
    ax.text(
        -0.02,                 # posizione orizzontale
        i,                     # posizione verticale
        exp,                   # nome esperimento
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=11
    )

ax.set_xlim(0, 1.1)
ax.legend()

# Valori alla fine delle barre
for i, (acc, f1) in enumerate(zip(accuracy_mb, f1_score_mb)):
    ax.text(acc + 0.01, i - bar_height/2, f"{acc:.3f}", va="center")
    ax.text(f1 + 0.01, i + bar_height/2, f"{f1:.3f}", va="center")

# Primo esperimento in alto
ax.invert_yaxis()

plt.tight_layout()
plt.show()