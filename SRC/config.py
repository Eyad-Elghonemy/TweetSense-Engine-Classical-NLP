import os
from dotenv import load_dotenv
import joblib
import gensim.downloader as api


# Load .env File
load_dotenv(override=True)

# Load environment Variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# SRC folder Path
SRC_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# Artifacts folder path
ARTIFACTS_FOLDER_PATH = os.path.join(SRC_FOLDER_PATH, 'artifacts')

# Load Models

# For BOW
bow_vectorizer = joblib.load(os.path.join(ARTIFACTS_FOLDER_PATH, 'bow_vecorizer.pkl'))
bow_model = joblib.load(os.path.join(ARTIFACTS_FOLDER_PATH, 'bbow_svc.pkl'))


# For Glove
glove_vectorizer = api.load("glove-twitter-25")
glove_model = joblib.load(os.path.join(ARTIFACTS_FOLDER_PATH, 'svc_glove_twitter.pkl'))

# Some Constants
EMOTICONS_MEANINGS = {
    ":)": "Happy",
    ":(": "Sad",
    ":D": "Very Happy",
    ":|": "Neutral",
    ":O": "Surprised",
    "<3": "Love",
    ";)": "Wink",
    ":P": "Playful",
    ":/": "Confused",
    ":*": "Kiss",
    ":')": "Touched",
    "XD": "Laughing",
    ":3": "Cute",
    ">:(": "Angry",
    ":-O": "Shocked",
    ":|]": "Robot",
    ":>": "Sly",
    "^_^": "Happy",
    "O_o": "Confused",
    ":-|": "Straight Face",
    ":X": "Silent",
    "B-)": "Cool",
    "<( '.' <)": "Dance",
    "(-_-)": "Bored",
    "(>_<)": "Upset",
    "(~_~)": "Sarcastic",
    "(o_o)": "Surprised",
    "(o.o)": "Shocked",
    ":0": "Shocked",
    ":*(": "Crying",
    ":v": "Pac-Man",
    "(^.^)v": "Double Victory",
    ":-D": "Big Grin",
    ":-*": "Blowing a Kiss",
    ":^)": "Nosey",
    ":-((": "Very Sad",
    ":-(": "Frowning",
}


# The Class To Predict Instead Of Just Number
SENTIMENT_MAPPING = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}