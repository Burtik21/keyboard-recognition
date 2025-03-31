import pandas as pd

# Definuj hlavičku (sloupce)
columns = [
    'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'max_intensity', 'dominant_freq',
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_13',
    'duration', 'keyboard'
]

# Vytvoř prázdný DataFrame s těmito sloupci
df = pd.DataFrame(columns=columns)

# Ulož do nového CSV
df.to_csv("data.csv", index=False)

print("✅ Vytvořen prázdný CSV s požadovanou strukturou.")
