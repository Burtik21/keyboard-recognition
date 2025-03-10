import pandas as pd

# Načtení obou datasetů
df_pressing = pd.read_csv("keyboard_recognition_data.csv")  # pressing = 1
df_no_pressing = pd.read_csv("keyboard_recognition_data_false.csv")  # pressing = 0

# Sloučení a zamíchání dat
df = pd.concat([df_pressing, df_no_pressing])
df = df.sample(frac=1).reset_index(drop=True)  # Zamíchat řádky

# Uložit nový dataset
df.to_csv("keyboard_final_data.csv", index=False)
print("Hotovo! Sloučený dataset uložen jako keyboard_final_data.csv")
