import matplotlib.pyplot as plt

# Data for each experiment
experiments = [
    {
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 15,
        "losses": [0.0012102470500394702, 0.0002628817455843091, 0.00614327983930707, 0.00011676184658426791, 0.0008621309534646571, 0.00010180610115639865, 6.224128446774557e-05, 8.294094004668295e-05, 4.068516500410624e-05, 6.280808156589046e-05, 6.15014650975354e-05, 4.940367944072932e-05, 5.307156243361533e-05, 3.601761636673473e-05, 0.007537702564150095],
        "test_loss": 0.0005395088746809051
    },
    {
        "learning_rate": 0.0005,
        "batch_size": 16,
        "epochs": 15,
        "losses": [0.0013340999139472842, 0.00033727538539096713, 0.003288397565484047, 0.00014266233483795077, 0.00013117417984176427, 0.00011383945093257353, 8.634002006147057e-05, 8.98079524631612e-05, 9.911584493238479e-05, 0.003155604237690568, 0.0076331403106451035, 5.6661636335775256e-05, 6.356366066029295e-05, 5.738671461585909e-05, 6.144719372969121e-05],
        "test_loss": 0.00046123316042212537
    },
    {
        "learning_rate": 8e-05,
        "batch_size": 16,
        "epochs": 15,
        "losses": [0.003464885987341404, 0.001349370344541967, 0.0030472232028841972, 0.0007803845219314098, 0.0007428104290738702, 0.008421576581895351, 0.0004830998368561268, 0.004464555066078901, 0.00041256373515352607, 0.00041962784598581493, 0.00030661452910862863, 0.0002654830168467015, 0.0002664592466317117, 0.0002668062224984169, 0.00028852722607553005],
        "test_loss": 0.001786579761756002
    },
    {
        "learning_rate": 8e-05,
        "batch_size": 64,
        "epochs": 15,
        "losses": [0.02030089497566223, 0.005483527202159166, 0.003454938530921936, 0.0027501608710736036, 0.003957122098654509, 0.0021265253890305758, 0.0025234618224203587, 0.0011463103583082557, 0.0009561291080899537, 0.0008443442056886852, 0.0019424690399318933, 0.0022813607938587666, 0.001878310926258564, 0.0024445122107863426, 0.0005819821381010115],
        "test_loss": 0.0005487005109898746
    },
    {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 15,
        "losses": [0.006299220956861973, 0.0029673564713448286, 0.002224308205768466, 0.001080700196325779, 0.002702833618968725, 0.00015297594654839486, 0.0017113765934482217, 0.0012991897528991103, 6.856314575998113e-05, 5.950733248027973e-05, 0.0007148683653213084, 0.00010733529779827222, 5.134259845362976e-05, 0.00016057855100370944, 4.0858631109585986e-05],
        "test_loss": 0.0003178933693561703
    }
]

# Plotting the loss curves
plt.figure(figsize=(10, 6))

for exp in experiments:
    plt.plot(range(1, exp["epochs"] + 1), exp["losses"], label=f'LR: {exp["learning_rate"]}, BS: {exp["batch_size"]}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)
plt.show()