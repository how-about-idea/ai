import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import csv
import requests
from bs4 import BeautifulSoup
import warnings

from TextGenerate import *
import math

class RLTextGenerator(TextGenerator):
    def text_generate(self, word):
        start_tokens = [self.word_to_index.get(word, 1)]
        num_tokens_generated = 0
        tokens_generated = [self.word_to_index.get(word, 1)]
        while num_tokens_generated < self.max_tokens:
            pad_len = self.max_tokens - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.max_tokens]
                sample_index = self.max_tokens - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])  # 주어진 문장
            y = self.model.predict([x, np.array([sample_index])])  # 주어진 문장의 다음 단어 예측값 y, _에 저장.
            sample_token = self.sample_from(y[0]) # y[0][다음 단어]로부터 다음 단어 가져오기
            if sample_token == 0 or sample_token == tokens_generated[sample_index] or y[0][sample_token] < 0:
                break
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in tokens_generated]
        )
        print(txt)
        return tokens_generated
        
    def text_generate_two_words(self, word1, word2):
        word1_text = self.text_generate(word1)               # word1로 생성된 문장
        word2_text = self.text_generate(word2)               # word2로 생성된 문장
        
        checkW1 = True                                       # word1다음에 word2가 생성될 때 True, 아닐 경우 False
        
        start_tokens = np.zeros((1, self.maxlen), dtype=np.int32)                  # 문장 생성
        start_tokens[0][0] = word1_text[0]
        text_generate = start_tokens.copy()                          # 최종 문장 생성 값
        y = self.model.predict([start_tokens, np.array([0])])
        max_logit = y[0][word2_text[0]]                    # 확률이 가장 높은 경우(초기값임)
 
        for i in range(1, len(word1_text)):
            start_tokens[0][i] = word1_text[i]
            y = self.model.predict([start_tokens, np.array([i])])           # 확률 구하기
            logit = y[0][word2_text[0]]                   # 확률 구하기
            if max_logit < logit:                             # 최대값의 text_generate로
                max_logit = logit 
                text_generate = start_tokens.copy()
                
        start_tokens = np.zeros((1, self.maxlen), dtype=np.int32)
        for i in range(len(word2_text)):
            start_tokens[0][i] = word2_text[i]
            y = self.model.predict([start_tokens, np.array([i])])
            logit = y[0][word1_text[0]]
            
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
            y = self.model.predict([text_generate, np.array([i])])  # 주어진 문장의 다음 단어 예측값 y, _에 저장.
            sample_token = self.sample_from(y[0]) # y[0][다음 단어]로부터 다음 단어 가져오기
            if sample_token == 0 or sample_token == text_generate[0][i] or y[0][sample_token] < 0:
                break
            text_generate[0][i + 1] = sample_token

        txt = " ".join(
            [self.detokenize(_) for _ in text_generate[0]]
        )

        txt = txt.replace(" ##", "")
        txt = txt.replace("[unk]", "")
        return txt
    
    def getPercentage(self, encoded):
        start_tokens = [self.word_to_index.get(word, 1) for word in encoded]
        
        percentage = []
        text = [0] * 30
        for idx in range(len(start_tokens) - 1):
            text[idx] = start_tokens[idx]

            logits = self.model.predict([np.array([text]), np.array([idx])])[0]
            logits, indices = tf.math.top_k(logits, k=1000, sorted=True)
            indices = np.asarray(indices).astype("int32")
            preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
            preds = np.asarray(preds).astype("float32")

            find = np.where(indices == start_tokens[idx + 1])[0]
            if len(find) == 0:
                percentage.append('-')
            else:
                percentage.append(math.floor(preds[find[0]] * 1000000) / 10000)

        return percentage