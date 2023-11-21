import pandas as pd
from tokenizers import BertWordPieceTokenizer

def getTokenizer():
  tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False)

  data_file = 'patent_title.txt'
  vocab_size = 1000000
  min_frequency = 5

  tokenizer.train(files=data_file,
                  vocab_size=vocab_size,
                  min_frequency=min_frequency)
  
  return tokenizer