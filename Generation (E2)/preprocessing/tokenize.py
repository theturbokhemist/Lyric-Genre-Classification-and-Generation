from transformers import AutoTokenizer
import contractions
import pickle
import re
import os
from tqdm import tqdm
import json

class Tokenizer:
    def __init__(self, model_name = "gpt2"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok_model_name = model_name
        self.SENT_BEGIN = "<s>"
        self.SENT_END = "</s>"
        self.word_count = {}
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0
    
    def fit(self, sentences, path = "Weights/tokenizer.json"):
        tokenized_sentences = self.tokenize(sentences, get_token_ids = False)["tokens"]
        for sent in tqdm(tokenized_sentences):
            for token in sent:
                if token not in self.word_count:
                    self.word_count[token] = 0
                self.word_count[token] += 1
        self.vocab_size = 0
        for word in self.word_count:
            self.word_index[word] = self.vocab_size
            self.index_word[self.vocab_size] = word
            self.vocab_size += 1

        directory = "/".join(path.split("/")[:-1])
        os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "tok_model_name": self.tok_model_name,
                "word_count": self.word_count,
                "word_index": self.word_index,
                "index_word": self.index_word,
                "vocab_size": self.vocab_size,
            }, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.tok_model_name = data["tok_model_name"]
            self.tok = AutoTokenizer.from_pretrained(self.tok_model_name)
            self.word_count = data["word_count"]
            self.word_index = data["word_index"]
            self.index_word = {int(k) : v for k,v in data["index_word"].items()}
            self.vocab_size = data["vocab_size"]


    def tokenize(self, sentences, get_token_ids = True):
        if(isinstance(sentences, list) == False):
            sentences = [sentences]
        sentences = [contractions.fix(s) for s in sentences]
        input_ids = self.tok(sentences).input_ids

        tokens = []
        for sent in input_ids:
            tokens.append([self.SENT_BEGIN] + [
                    re.sub(r'[\s]', '_', self.tok.decode(i)) # replace whitespaces with underscores. Word2Vec has trouble embedding words with spaces
                    for i in sent
                ] + [self.SENT_END])
        
        ret = {"tokens": tokens}
        # print(tokens)
        if get_token_ids:
            ret["token_ids"] = [
                [self.word_index[w] for w in sent]
                for sent in tokens
            ]
        
        return ret
    
    def decode(self, token_ids, remove_special_tokens = True):
        if(isinstance(token_ids, list) == False):
            token_ids = [token_ids]
        if(isinstance(token_ids[0], list) == False):
            token_ids = [token_ids]
        
        sentences = []
        for sent in token_ids:
            sentences.append("".join(self.index_word[i] for i in sent))
        if remove_special_tokens:
            sentences = [
                s.replace(self.SENT_BEGIN, "").replace(self.SENT_END, "") 
                for s in sentences
            ]
            
        sentences = [
            s.replace(" ", "").replace("_", " ") 
            for s in sentences
        ]
        
        return sentences
            
    