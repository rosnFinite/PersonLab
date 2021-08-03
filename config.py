
class config:

    IMG_SHAPE = (400,400,3)

    NUM_KP = 17

    KP_RADIUS = 32

    KP_NAMES = ["Nase",#0
               "Schulter(r)",#1
               "Ellenbogen(r)",#2
               "Handgelenk(r)",#3
               "Schulter(l)",#4
               "Ellenbogen(l)",#5
               "Handgelenk(l)",#6
               "Hüfte(r)",#7
               "Knie(r)",#8
               "Fußgelenk(r)",#9
               "Hüfte(l)",#10
               "Knie(l)",#11
               "Fußgelenk(l)",#12
               "Auge(r)",#13
               "Auge(l)",#14
               "Ohr(r)",#15
               "Ohr(l)"]#16

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

    BATCH_SIZE = 4

    PEAK_THRESH = 0.001

    INSTANCE_SEG_THRESH = 0.25

