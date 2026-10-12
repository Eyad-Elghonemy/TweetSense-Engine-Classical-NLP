from SRC.utils.text_processor import TextProcessor
from SRC.config import bow_vectorizer, bow_model, SENTIMENT_MAPPING, glove_vectorizer, glove_model
from typing import List, Dict
import numpy as np


class TextClassifier:
    
    def __init__(self):
        
        self.processor = TextProcessor()
        self.vectorizer = bow_vectorizer
        self.model = bow_model
        self.sentiment_mapping = SENTIMENT_MAPPING
        
        
    def predict(self, texts: List[str]) -> List[Dict[str, str]]:
            
        # Clean Text
        cleaned_text = [self.processor.process_text(text) for text in texts]
        
        # Vectorize Text
        vectors = self.vectorizer.transform(cleaned_text).toarray()
        
        # Predict Sentiment
        raw_predictions = self.model.predict(vectors)
        
        # Append Here
        predictions = []
        for text, label in zip(texts, raw_predictions):
            sentiment = self.sentiment_mapping[int(label)]
            predictions.append({
                "text": text,
                "sentiment": sentiment
            })
            
        return predictions
    


class TextClassifierGlove: 
    
    def __init__(self):
            
            self.processor = TextProcessor()
            self.vectorizer = glove_vectorizer
            self.model = glove_model
            self.sentiment_mapping = SENTIMENT_MAPPING
            
            
    def _sentence_to_vector_pretrained(self, sentence):
        
        words = sentence.lower().split()
        words_vectors = [self.vectorizer[word] for word in words if word in self.vectorizer]
        
        if len(words_vectors) == 0:
            return np.zeros(25)
        
        return np.mean(words_vectors, axis=0)
    
       
    def predict(self, texts: List[str]) -> List[Dict[str, str]]:
         
         
        # Clean Text
        cleaned_text = [self.processor.process_text(text) for text in texts]
        
        # Vectorize Text
        vectors = np.array([self._sentence_to_vector_pretrained(text) for text in cleaned_text])
        
        # Predict Sentiment
        raw_predictions = self.model.predict(vectors)
        
        # Append Here
        predictions = []
        for text, label in zip(texts, raw_predictions):
            sentiment = self.sentiment_mapping[int(label)]
            predictions.append({
                "text": text,
                "sentiment": sentiment
            })
            
        return predictions