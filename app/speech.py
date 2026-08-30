import pygame
import threading
import os

# =====================================================
# Initialize Pygame Mixer
# =====================================================

pygame.mixer.init()

# =====================================================
# Welcome Audio
# =====================================================

AUDIO_FILE = os.path.join(
    "assets",
    "welcome.mp3"
)

# =====================================================
# Play Audio
# =====================================================

def _play_audio():

    # Don't start another greeting if one is already playing
    if pygame.mixer.music.get_busy():
        return

    pygame.mixer.music.load(AUDIO_FILE)
    pygame.mixer.music.play()

# =====================================================
# Public Function
# =====================================================

def speak():

    threading.Thread(
        target=_play_audio,
        daemon=True
    ).start()