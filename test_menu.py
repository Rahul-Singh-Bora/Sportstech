import sys
sys.path.insert(0, '.')
print("About to import...")
from vidio_track import GAMES, choose_game
print("Imported successfully!")
print("Games available:", list(GAMES.keys()))
print("\nNow calling choose_game()...")
