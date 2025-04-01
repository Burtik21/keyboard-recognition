# config.py
#globalni
BASE_URL= "http://sajmiho.lol/keyboard"

# Zvukové nastavení
SAMPLE_RATE = 44100       # Hz
CHANNELS = 1              # mono
DURATION = 1.0            # délka nahrávacího okna v sekundách
FRAME_LENGTH = 1024       # počet vzorků v rámci jednoho okna (např. pro RMS)
HOP_LENGTH = 256          # kolik vzorků posunout při každém výpočtu

# Detekce
WINDOW_DURATION = 0.5
RMS_THRESHOLD = 0.0012    # práh pro RMS detekci
COOLDOWN_SECONDS = 0.2    # minimální čas mezi dvěma detekcemi

# Logování
VERBOSE = True
