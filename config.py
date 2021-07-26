
class config:

    IMG_SHAPE = (400,400,3)

    NUM_KP = 17

    KP_RADIUS = 32

    KP_NAMES = ["Nase",
               "Schulter(r)",
               "Ellenbogen(r)",
               "Handgelenk(r)",
               "Schulter(l)",
               "Ellenbogen(l)",
               "Handgelenk(l)",
               "Hüfte(r)",
               "Knie(r)",
               "Fußgelenk(r)",
               "Hüfte(l)",
               "Knie(l)",
               "Fußgelenk(l)",
               "Auge(r)",
               "Auge(l)",
               "Ohr(r)",
               "Ohr(l)"]

    EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3)
    ]

    LOSS_WEIGHTS = {
        'heatmap': 4,
        'seg': 2,
        'short': 1,
        'mid': 0.4,
        'long': 0.1
    }

    TRAINING = True

    NUM_EPOCHS = 1

    BATCH_SIZE = 8