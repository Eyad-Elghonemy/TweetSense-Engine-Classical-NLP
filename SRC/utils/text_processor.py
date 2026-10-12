import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from SRC.config import EMOTICONS_MEANINGS

class TextProcessor:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.emoticons_meanings = EMOTICONS_MEANINGS
        
    def _remove_pattern(self, input_txt, pattern):
        return re.sub(pattern, '', input_txt)
    
    def _remove_excessive_chars(self, input_str, max_repeats=2):
        pattern = f"(\\w)\\1{{{max_repeats},}}"
        return re.sub(pattern, r"\1", input_str)
        
    def _convert_emoticons(self, text: str):
        for emotion, meaning in self.emoticons_meanings.items():
            text = text.replace(emotion, meaning)
        return text
        
    def _remove_redundant_words(self, text: str):
        contraction_pattern = re.compile(r"'\w+|\w+'\w+|\w+'")
        text = contraction_pattern.sub('', text)
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def _lemmatize_text(self, text: str): 
        words = text.split()
        lemmatize_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatize_words)
    
    def process_text(self, text: str):
        text = self._remove_pattern(text, r"@[\w]*")
        text = self._remove_pattern(text, r"https?://\S+|www\.\S+")
        text = self._remove_excessive_chars(text)
        text = self._convert_emoticons(text)
        text = re.sub(r"[^a-zA-Z#]", " ", text)
        text = " ".join([w for w in text.split() if len(w) > 3])
        text = self._remove_pattern(text, r"(?<=\w)\d+|\d+(?=\w)")
        text = self._remove_pattern(text, r"[!@#$%^&*()_+{}\[\]:;<>,.?~\\\/]")
        text = self._remove_redundant_words(text)
        text = self._lemmatize_text(text)
        return text
