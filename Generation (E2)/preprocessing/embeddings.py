import os
from tqdm.auto import tqdm
import numpy as np
from gensim.models import Word2Vec
from gensim import models

class Embedding:
    def train(
            self, path, sentences, 
            embeddings_size = 300, window = 5, min_count = 1, sg = 1
        ):
        directory = "/".join(path.split("/")[:-1])
        os.makedirs(directory, exist_ok=True)
        self.path = path
        print(f"training word2vec model with {len(sentences)} sentences")
        self.model = Word2Vec(
            sentences = sentences,
            vector_size = embeddings_size,
            window = window, min_count = min_count, sg = sg
        )
        print(f"finished training >> saving to {self.path}")
        self.model.wv.save_word2vec_format(self.path, binary=False)
        self.load(self.path)

    def load(self, path):
        self.path = path
        self.model = models.KeyedVectors.load_word2vec_format(self.path, binary = False)
        self.vocab_size = len(self.model)
        self.embeddings_size = self.model.vector_size
    
