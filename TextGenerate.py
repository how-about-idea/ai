import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import csv
import requests
from bs4 import BeautifulSoup
import warnings

def getVocab():
    vocab = []
    with open('vocab.csv', 'r', encoding = 'UTF-8') as f:
        reader = csv.reader(f)
        for i in reader:
            vocab = i
            break
    return vocab

def getword_to_index(vocab):
    word_to_index = {}
    for index, word in enumerate(vocab):
      word_to_index[word] = index
    return word_to_index

class TextGenerator():
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(self, max_tokens, index_to_word, word_to_index, model, top_k=20, maxlen=30):
        self.max_tokens = max_tokens
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        self.model = model
        self.k = top_k
        self.maxlen = maxlen

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]
        
    def text_generate(self, word):
        start_tokens = [self.word_to_index.get(word, 1)]
        num_tokens_generated = 0
        tokens_generated = [self.word_to_index.get(word, 1)]
        while num_tokens_generated < self.max_tokens:
            pad_len = self.maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.maxlen]
                sample_index = self.maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])  # 주어진 문장
            y, _ = self.model.predict(x)  # 주어진 문장의 다음 단어 예측값 y, _에 저장.
            sample_token = self.sample_from(y[0][sample_index]) # y[0][다음 단어]로부터 다음 단어 가져오기
           
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        return tokens_generated
        
    def text_generate_two_words(self, word1, word2):
        word1_text = self.text_generate(word1)               # word1로 생성된 문장
        word2_text = self.text_generate(word2)               # word2로 생성된 문장
        
        checkW1 = True                                       # word1다음에 word2가 생성될 때 True, 아닐 경우 False
        
        start_tokens = np.zeros((1, self.maxlen), dtype=np.int32)                  # 문장 생성
        start_tokens[0][0] = word1_text[0]
        text_generate = start_tokens.copy()                          # 최종 문장 생성 값
        y, _ = self.model.predict(start_tokens)
        max_logit = y[0][0][word2_text[0]]                    # 확률이 가장 높은 경우(초기값임)
 
        for i in range(1, self.maxlen):
            if word1_text[i] == 0:                          # word1_text가 끝날 경우 break
                break
                
            start_tokens[0][i] = word1_text[i]
            y, _ = self.model.predict(start_tokens)           # 확률 구하기
            logit = y[0][i][word2_text[0]]                   # 확률 구하기
            if max_logit < logit:                             # 최대값의 text_generate로
                max_logit = logit 
                text_generate = start_tokens.copy()
                
        start_tokens = np.zeros((1, self.maxlen), dtype=np.int32)
        for i in range(self.maxlen):
            if word2_text[i] == 0:
                break
                
            start_tokens[0][i] = word2_text[i]
            y, _ = self.model.predict(start_tokens)
            logit = y[0][i][word1_text[0]]
            
            if max_logit < logit:
                max_logit = logit
                text_generate = start_tokens.copy()
                checkW1 = False
               
        idx = 0
        for i in range(self.maxlen):
            if text_generate[0][i] == 0:
                if checkW1:
                    text_generate[0][i] = word2_text[0]
                else:
                    text_generate[0][i] = word1_text[0]
                idx = i
                break
                
        for i in range(idx, self.maxlen - 1):
            y, _ = self.model.predict(text_generate)  # 주어진 문장의 다음 단어 예측값 y, _에 저장.
            sample_token = self.sample_from(y[0][i]) # y[0][다음 단어]로부터 다음 단어 가져오기
            text_generate[0][i + 1] = sample_token

        txt = " ".join(
            [self.detokenize(_) for _ in text_generate[0]]
        )

        txt = txt.replace(" ##", "")
        txt = txt.replace("[unk]", "")
        return txt
    
    def relationPatent(self, txt):
        url = 'http://plus.kipris.or.kr/openapi/rest/patUtiModInfoSearchSevice/freeSearchInfo?word=' + txt + '&patent=true&utility=true&docsStart=1&docsCount=20&accessKey=vsl33aAYpn21YUazLEhqkhZXW3/VTcs7bYCq3D4aEGk='
        response = requests.get(url)

        warnings.filterwarnings("ignore", category=UserWarning)
        soup = BeautifulSoup(response.text, "lxml")
        RelationPatent = soup.find_all('inventionname')
        rp = []
        for title in RelationPatent:
            if '다온상표' not in title or '1833-8891' not in title:
                rp.append(str(title)[15:-16])

        rp = list(set(rp))
        for i in range(5 - len(rp)):
            rp.append("-")

        return rp[:5]