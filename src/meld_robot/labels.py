"""MELD label spaces (fixed order: used by every model head and metric)."""

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]

EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
SENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}

# Coarse sentiment implied by each emotion (used for the state JSON when a sentiment head is absent).
EMOTION_TO_SENTIMENT = {
    "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
    "joy": "positive", "surprise": "neutral", "neutral": "neutral",
}

NUM_EMOTIONS = len(EMOTIONS)
NUM_SENTIMENTS = len(SENTIMENTS)
