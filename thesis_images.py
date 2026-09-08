import numpy as np
import matplotlib.pyplot as plt

'''
# ============================================================
# CONFIGURAZIONE
# ============================================================

BLUE = "#293B8F"
RED = "#A82B2B"
BLACK = "#111111"
YELLOW = "#FFF36A"

POINT_SIZE = 50


# ============================================================
# FUNZIONE PER DISEGNARE I PUNTI
# ============================================================

def draw_points(ax, blue_points, red_points):
    """Disegna le due classi di punti."""

    blue_points = np.array(blue_points)
    red_points = np.array(red_points)

    ax.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        s=POINT_SIZE,
        c=BLUE,
        #edgecolors=BLACK,
        #linewidths=1.5,
        #zorder=5
    )

    ax.scatter(
        red_points[:, 0],
        red_points[:, 1],
        s=POINT_SIZE,
        c=RED,
        #edgecolors=BLACK,
        #linewidths=1.5,
        #zorder=5
    )


# ============================================================
# PRIMA IMMAGINE
# ============================================================

def create_first_image():

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    # --------------------------------------------------------
    # Punti blu
    # --------------------------------------------------------

    blue_points = [
        (2.05, 3.55),
        (2.65, 3.30),
        (3.25, 2.98),
        (2.35, 2.82),
        (2.72, 2.82),
        (3.25, 2.30),
        (2.70, 1.92),
        (3.02, 1.50),
    ]

    # --------------------------------------------------------
    # Punti rossi
    # --------------------------------------------------------

    red_points = [
        (0.72, 2.25),
        (1.15, 2.55),
        (1.50, 1.98),
        (0.72, 1.42),
        (1.15, 1.78),
        (1.72, 1.25),
        (2.22, 0.85),
    ]

    draw_points(ax, blue_points, red_points)

    # --------------------------------------------------------
    # Retta w·x + b = 0
    #
    # y = m*x + q
    # --------------------------------------------------------

    m = -0.82
    q = 3.95

    x = np.linspace(0.45, 3.35, 300)
    y = m * x + q

    ax.plot(
        x,
        y,
        color=BLACK,
        linewidth=2.5,
        solid_capstyle="butt",
        zorder=3
    )

    # --------------------------------------------------------
    # Equazione
    # --------------------------------------------------------

    ax.text(
        3.12,
        0.63,
        r"w·x + b = 0",
        fontsize=19,
        color=BLACK,
        ha="left",
        va="center"
    )

    # --------------------------------------------------------
    # Aspetto
    # --------------------------------------------------------

    ax.set_xlim(0.25, 4.05)
    ax.set_ylim(0.20, 4.05)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0)

    plt.savefig(
        "images/classificatore_lineare.pdf",
        #dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ============================================================
# SECONDA IMMAGINE — SVM
# ============================================================

def create_second_image():

    fig, ax = plt.subplots(figsize=(10.1, 4.8))

    # --------------------------------------------------------
    # Punti blu
    # --------------------------------------------------------

    blue_points = [
        (4.15, 4.05),
        (4.80, 3.80),
        (5.55, 3.48),
        (4.45, 3.28),
        (5.05, 3.28),
        (5.55, 2.72),
        (4.95, 2.20),
    ]

    # --------------------------------------------------------
    # Punti rossi
    # --------------------------------------------------------

    red_points = [
        (3.25, 3.05),
        (2.72, 2.75),
        (3.62, 2.48),
        (2.72, 2.02),
        (3.82, 1.85),
        (2.72, 1.62),
        (4.25, 1.30),
    ]

    draw_points(ax, blue_points, red_points)

    # --------------------------------------------------------
    # Retta centrale
    #
    # w·x + b = 0
    # --------------------------------------------------------

    m = -1.70
    q = 9.95

    x = np.linspace(2.65, 5.75, 300)

    y_center = m * x + q

    # --------------------------------------------------------
    # Margini
    # --------------------------------------------------------

    # Distanza verticale usata solo per costruire
    # graficamente i due margini.
    delta = 0.58

    y_upper = y_center + delta
    y_lower = y_center - delta

    # --------------------------------------------------------
    # Margine superiore
    # --------------------------------------------------------

    ax.plot(
        x,
        y_upper,
        color=BLACK,
        linewidth=2.3,
        linestyle=(0, (5, 5)),
        solid_capstyle="butt",
        zorder=2
    )

    # --------------------------------------------------------
    # Retta centrale
    # --------------------------------------------------------

    ax.plot(
        x,
        y_center,
        color=BLACK,
        linewidth=2.5,
        solid_capstyle="butt",
        zorder=3
    )

    # --------------------------------------------------------
    # Margine inferiore
    # --------------------------------------------------------

    ax.plot(
        x,
        y_lower,
        color=BLACK,
        linewidth=2.3,
        linestyle=(0, (5, 5)),
        solid_capstyle="butt",
        zorder=2
    )

    # ========================================================
    # EQUAZIONE DELLA RETTA CENTRALE
    # ========================================================

    ax.text(
        0.95,
        3.98,
        r"w·x + b = 0",
        fontsize=21,
        color=BLACK
    )

    # ========================================================
    # EQUAZIONE DEL MARGINE INFERIORE
    # ========================================================

    ax.text(
        2.15,
        0.55,
        r"w·x + b = -1",
        fontsize=21,
        color=BLACK
    )

    # ========================================================
    # EQUAZIONE DEL MARGINE SUPERIORE
    # ========================================================

    ax.text(
        5.72,
        1.08,
        r"w·x + b = +1",
        fontsize=21,
        color=BLACK,
        ha="left",
        va="center"
    )

    # ========================================================
    # FRECCIA DEL "MARGIN"
    # ========================================================

    # Punto centrale della freccia
    x_arrow = 4.78

    yc = m * x_arrow + q
    yu = yc + delta
    yl = yc - delta

    # Freccia
    ax.annotate(
        "",
        xy=(x_arrow + 0.12, yu - 0.02),
        xytext=(x_arrow - 0.12, yl + 0.02),
        arrowprops=dict(
            arrowstyle="<->",
            color=RED,
            linewidth=2.5,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=10
    )

    # --------------------------------------------------------
    # Scritta "margin"
    # --------------------------------------------------------

    ax.text(
        x_arrow + 0.35,
        yc + 0.12,
        "margin",
        fontsize=23,
        fontweight="bold",
        color=RED,
        ha="left",
        va="center"
    )

    # ========================================================
    # ASPETTO GRAFICO
    # ========================================================

    ax.set_xlim(0.45, 8.05)
    ax.set_ylim(0.20, 4.55)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0)

    plt.savefig(
        "images/svm_margine.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ============================================================
# ESECUZIONE
# ============================================================

create_first_image()
create_second_image()
'''


# ============================================================
# STILE
# ============================================================

BLUE = "#293B8F"
RED = "#A82B2B"
BLACK = "#111111"
YELLOW = "#FFF36A"

POINT_SIZE = 220


# ============================================================
# FUNZIONE PER DISEGNARE I PUNTI
# ============================================================

def draw_points(ax, blue_points, red_points):
    """
    Disegna i punti delle due classi.
    """

    blue_points = np.asarray(blue_points)
    red_points = np.asarray(red_points)

    ax.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        s=POINT_SIZE,
        color=BLUE,
        edgecolor=BLACK,
        linewidth=1.5,
        zorder=10
    )

    ax.scatter(
        red_points[:, 0],
        red_points[:, 1],
        s=POINT_SIZE,
        color=RED,
        edgecolor=BLACK,
        linewidth=1.5,
        zorder=10
    )


# ============================================================
# PRIMA IMMAGINE
# ============================================================

def create_first_image():

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    # --------------------------------------------------------
    # Punti blu
    # --------------------------------------------------------

    blue_points = [
        (2.05, 3.55),
        (2.65, 3.30),
        (3.25, 2.98),
        (2.35, 2.82),
        (2.72, 2.82),
        (3.25, 2.30),
        (2.70, 1.92),
        (3.02, 1.50),
    ]

    # --------------------------------------------------------
    # Punti rossi
    # --------------------------------------------------------

    red_points = [
        (0.72, 2.25),
        (1.15, 2.55),
        (1.50, 1.98),
        (0.72, 1.42),
        (1.15, 1.78),
        (1.72, 1.25),
        (2.22, 0.85),
    ]

    # --------------------------------------------------------
    # Retta di separazione
    #
    # y = m*x + q
    # --------------------------------------------------------

    m = -0.82
    q = 3.95

    x = np.linspace(0.45, 3.35, 300)
    y = m * x + q

    ax.plot(
        x,
        y,
        color=BLACK,
        linewidth=2.5,
        zorder=3
    )

    # Punti sopra la retta
    draw_points(ax, blue_points, red_points)

    # --------------------------------------------------------
    # Equazione
    # --------------------------------------------------------

    ax.text(
        3.10,
        0.63,
        r"$\mathbf{w}\cdot\mathbf{x}+b=0$",
        fontsize=19,
        color=BLACK,
        ha="left",
        va="center"
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    ax.set_xlim(0.25, 4.05)
    ax.set_ylim(0.20, 4.05)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0)

    plt.savefig(
        "classificatore_lineare.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ============================================================
# SECONDA IMMAGINE — SVM
# ============================================================

def create_second_image():

    fig, ax = plt.subplots(figsize=(10.1, 4.8))

    # ========================================================
    # DEFINIZIONE DELLA RETTA SVM
    # ========================================================

    # Retta:
    #
    #     y = m*x + q
    #
    # equivalente a:
    #
    #     w1*x + w2*y + b = 0
    #
    # con:
    #
    #     w = (-m, 1)
    #     b = -q
    #
    # ========================================================

    m = -1.70
    q = 9.95

    # Vettore normale alla retta
    w = np.array([-m, 1.0])

    # Norma di w
    w_norm = np.linalg.norm(w)

    # Vettore normale unitario
    w_hat = w / w_norm

    # b della retta centrale
    b = -q

    # ========================================================
    # FUNZIONE PER LE TRE RETTE
    # ========================================================

    def line_y(x, value):
        """
        Restituisce y per:

            w·x + b = value

        dove value = 0, +1 oppure -1.
        """

        # w1*x + w2*y + b = value
        #
        # => y = (value - w1*x - b) / w2

        return (value - w[0] * x - b) / w[1]

    # ========================================================
    # DOMINIO DELLE RETTE
    # ========================================================

    x = np.linspace(2.55, 5.75, 400)

    y_center = line_y(x, 0)
    y_upper = line_y(x, +1)
    y_lower = line_y(x, -1)

    # ========================================================
    # DISEGNO DELLE RETTE
    # ========================================================

    # Margine inferiore
    ax.plot(
        x,
        y_lower,
        color=BLACK,
        linewidth=2.3,
        linestyle=(0, (5, 5)),
        zorder=2
    )

    # Retta centrale
    ax.plot(
        x,
        y_center,
        color=BLACK,
        linewidth=2.5,
        zorder=3
    )

    # Margine superiore
    ax.plot(
        x,
        y_upper,
        color=BLACK,
        linewidth=2.3,
        linestyle=(0, (5, 5)),
        zorder=2
    )

    # ========================================================
    # PUNTI
    #
    # I punti di supporto sono costruiti IN MODO CHE
    # appartengano esattamente ai margini:
    #
    #     w·x+b = +1
    #     w·x+b = -1
    #
    # ========================================================

    # --------------------------------------------------------
    # Punti blu
    # --------------------------------------------------------

    blue_points = [
        (4.15, 4.05),
        (4.80, 3.80),
        (5.55, 3.48),
        (4.45, 3.28),
        (5.05, 3.28),
        (5.55, 2.72),
        (4.95, 2.20),
    ]

    # --------------------------------------------------------
    # Punti rossi
    # --------------------------------------------------------

    red_points = [
        (3.25, 3.05),
        (2.72, 2.75),
        (3.62, 2.48),
        (2.72, 2.02),
        (3.82, 1.85),
        (2.72, 1.62),
        (4.25, 1.30),
    ]

    # --------------------------------------------------------
    # Modifichiamo alcuni punti affinché siano ESATTAMENTE
    # sui margini.
    # --------------------------------------------------------

    # Support vector BLU:
    # deve stare su w·x+b = +1

    x_blue_sv = 4.95
    y_blue_sv = line_y(x_blue_sv, +1)

    # Sostituiamo l'ultimo punto blu
    blue_points[-1] = (x_blue_sv, y_blue_sv)

    # Support vector ROSSO:
    # deve stare su w·x+b = -1

    x_red_sv = 4.25
    y_red_sv = line_y(x_red_sv, -1)

    # Sostituiamo l'ultimo punto rosso
    red_points[-1] = (x_red_sv, y_red_sv)

    # --------------------------------------------------------
    # Disegna i punti DOPO le rette.
    #
    # In questo modo la retta tratteggiata viene nascosta
    # sotto il punto e il punto appare "appoggiato" al margine.
    # --------------------------------------------------------

    draw_points(ax, blue_points, red_points)

    # ========================================================
    # EQUAZIONE DELLA RETTA CENTRALE
    # ========================================================

    ax.text(
        0.95,
        3.98,
        r"$\mathbf{w}\cdot\mathbf{x}+b=0$",
        fontsize=21,
        color=BLACK,
        ha="left",
        va="center"
    )

    # ========================================================
    # EQUAZIONE MARGINE INFERIORE
    # ========================================================

    ax.text(
        2.15,
        0.55,
        r"$\mathbf{w}\cdot\mathbf{x}+b=-1$",
        fontsize=21,
        color=BLACK,
        bbox=dict(
            facecolor=YELLOW,
            edgecolor="none",
            pad=3
        )
    )

    # ========================================================
    # EQUAZIONE MARGINE SUPERIORE
    # ========================================================

    ax.text(
        5.72,
        1.08,
        r"$\mathbf{w}\cdot\mathbf{x}+b=+1$",
        fontsize=21,
        color=BLACK,
        ha="left",
        va="center",
        bbox=dict(
            facecolor=YELLOW,
            edgecolor="none",
            pad=3
        )
    )

    # ========================================================
    # MARGIN
    #
    # La freccia deve:
    #
    #   1. essere ortogonale alle rette;
    #   2. partire dalla retta centrale;
    #   3. arrivare alla retta superiore.
    #
    # Usiamo quindi direttamente il vettore normale w_hat.
    # ========================================================

    # Punto sulla retta centrale
    P0 = np.array([4.55, line_y(4.55, 0)])

    # La distanza tra:
    #
    #     w·x+b=0
    #
    # e
    #
    #     w·x+b=+1
    #
    # è:
    #
    #             1 / ||w||
    #
    # quindi il punto sulla retta +1 è:
    #
    #     P1 = P0 + w_hat / ||w||
    #
    P1 = P0 + w_hat / w_norm

    # --------------------------------------------------------
    # Doppia freccia
    # --------------------------------------------------------

    ax.annotate(
        "",
        xy=P1,
        xytext=P0,
        arrowprops=dict(
            arrowstyle="<->",
            color=RED,
            linewidth=2.5,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=20
    )

    # ========================================================
    # SCRITTA "margin"
    #
    # Posizionata accanto alla doppia freccia.
    # ========================================================

    # Spostamento laterale rispetto alla freccia.
    # Usiamo la direzione della retta per evitare che il testo
    # finisca sopra la freccia.

    line_direction = np.array([1.0, m])
    line_direction = line_direction / np.linalg.norm(line_direction)

    text_position = (
        (P0 + P1) / 2
        + 0.32 * line_direction
    )

    ax.text(
        text_position[0],
        text_position[1],
        "margin",
        fontsize=23,
        fontweight="bold",
        color=RED,
        ha="left",
        va="center"
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    ax.set_xlim(0.45, 8.05)
    ax.set_ylim(0.20, 4.55)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0)

    plt.savefig(
        "svm_margine.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ============================================================
# ESECUZIONE
# ============================================================

create_first_image()
create_second_image()
