# word_prediction.py

from collections import Counter

# Extended word list – you can add more based on common usage
COMMON_WORDS = [
    "HELLO", "HELP", "PLEASE", "THANK", "YES", "NO", "LOVE", "HAPPY", "NAME", "WHAT", 
    "WHO", "WHY", "WHERE", "WHEN", "HOW", "SIGN", "LANGUAGE", "YOU", "ME", "GOOD", 
    "BAD", "SORRY", "WATER", "FOOD", "FINE", "FRIEND", "TODAY", "TOMORROW", "YESTERDAY",
    "SCHOOL", "HOME", "MOTHER", "FATHER", "BROTHER", "SISTER", "FAMILY", "WORK", "TIRED",
    "BATHROOM", "TEACHER", "STUDENT", "LOVE", "PEACE", "STOP", "GO", "RIGHT", "LEFT",
    "FAST", "SLOW", "BIG", "SMALL", "CAT", "DOG", "SUN", "MOON", "RAIN", "HOT", "COLD",
    "READY", "WAIT", "AGAIN", "UNDERSTAND", "QUESTION", "ANSWER", "BEAUTIFUL", "DIFFERENT"
]

def predict_word(letter_sequence):
    """Return a list of possible matching words from the dictionary."""
    matches = []
    letter_sequence = letter_sequence.upper()
    for word in COMMON_WORDS:
        if word.startswith(letter_sequence):
            matches.append(word)
    
    # Return most likely matches first
    return sorted(matches, key=lambda w: (len(w), w))
