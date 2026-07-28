import pygame
import threading
import os

# Initialize pygame mixer
pygame.mixer.init()

# Audio file path
AUDIO_FILE = os.path.join("assets", "welcome.mp3")


def _play_audio():
    pygame.mixer.music.load(AUDIO_FILE)
    pygame.mixer.music.play()


def speak():
    threading.Thread(
        target=_play_audio,
        daemon=True
    ).start()