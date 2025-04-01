from pynput import keyboard
import time

print("Mačkej klávesy... (pro ukončení stiskni ESC)")

last_time = None


def on_press(key):
    global last_time

    now = time.time()

    if key == keyboard.Key.esc:
        print("Ukončeno.")
        return False  # zastaví posluchač

    if last_time is not None:
        interval = now - last_time
        print(f"⏱️ Interval: {interval:.4f} sekund")

    last_time = now


# Spuštění posluchače
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
