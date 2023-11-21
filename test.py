from flask import Flask, jsonify, request
from flask_cors import CORS
import keras
import numpy as np
import csv
from tabulate import tabulate

app = Flask(__name__)
CORS(app) # React 연동 간 CORS 에러 해결

from Layers import *
from TextGenerate import *
from RLTextGenerate import *
from tokenizer import *
from DQN import *

model = tf.keras.models.load_model('./patent_final.h5', compile = False, custom_objects= {'TokenAndPositionEmbedding':TokenAndPositionEmbedding, 'TransformerBlock':TransformerBlock})

# modelRL = tf.keras.models.load_model('RLModel44.h5', compile = False) # 기존 모델(40만 문장 학습)
# modelRL = tf.keras.models.load_model('RLModel_V2_7.h5', compile = False) # 신규 모델(7만 문장 학습)
# modelRL = tf.keras.models.load_model('RLModel_V2_17.h5', compile = False)  # 신규 모델(17만 문장 학습)
# modelRL = tf.keras.models.load_model('RLModel_V2_40.h5', compile = False)  # 신규 모델(40만 문장 학습)
modelRL = tf.keras.models.load_model('RLModel_V2_49.h5', compile = False)  # 신규 모델(40만 문장 학습)

vocab = getVocab()
word_to_index = getword_to_index(vocab)

num_tokens_generated = 30
text_gen = TextGenerator(num_tokens_generated, vocab, word_to_index, model)

# result = text_gen.text_generate_two_words("전류", "전압")
# print(result)

# text_gen = text_gen.text_generate_two_words("무인", "비행기")
# print(text_gen)

text_gen_RL = RLTextGenerator(num_tokens_generated, vocab, word_to_index, modelRL)
# Test Data
# text_gen_RL_1 = text_gen_RL.text_generate_two_words("전류", "전압")
# print(text_gen_RL_1)

#KIPRIS API 호출_연관 특허 출력
# relList = text_gen.relationPatent(text_gen_RL_1)


######################################################################
# 인공지능 재학습 코드                                                #
######################################################################
tok = getTokenizer()
dqn = DQN(modelRL, model, word_to_index)

@app.route('/api/modelUpdate', methods=['POST'])
def modelUpdate():
    # 1. JSON 데이터 받아오기(POST 방식)_[type='dict']
    request_json = request.get_json()
    print(request_json)

    # 2. dict 데이터 읽어내기
    sentence = request_json['sentence']  # 해당 문장
    reward = request_json['userStarRating']    # 개인 별점
    combineWord1 = request_json['combineWord1'] # 조합 단어 1
    combineWord2 = request_json['combineWord2'] # 조합 단어 2

    train_reward = None
    print("reward: ", reward)
    if reward == 1:
        train_reward = -1
    elif reward == 2:
        train_reward = -0.5
    elif reward == 3:
        train_reward = 0
    elif reward == 4:
        train_reward = 0.5
    elif reward == 5:
        train_reward = 1
        
    encoded = tok.encode(sentence)
    prev_percentage = text_gen_RL.getPercentage(encoded.tokens)
    df_columns = encoded.tokens[1:]
    df = pd.DataFrame(columns=df_columns)
    df.loc[0] = prev_percentage


    # 재학습 시연 결과용    
    # for i in range(20):
    #     dqn.train(encoded.tokens, train_reward)

    #     text_gen_RLV2 = RLTextGenerator(num_tokens_generated, vocab, word_to_index, dqn.model)

    #     curr_percentage = text_gen_RLV2.getPercentage(encoded.tokens)

    #     # result = text_gen.text_generate_two_words(combineWord1, combineWord2) # 재학습 모델 문장 생성 처리
    #     # print("재학습 모델 생성 문장:", result)

    #     print(curr_percentage)
    #     df.loc[i + 1] = curr_percentage

    # 원래 결과
    dqn.train(encoded.tokens, train_reward)

    text_gen_RLV2 = RLTextGenerator(num_tokens_generated, vocab, word_to_index, dqn.model)

    curr_percentage = text_gen_RLV2.getPercentage(encoded.tokens)

    # result = text_gen.text_generate_two_words(combineWord1, combineWord2) # 재학습 모델 문장 생성 처리
    # print("재학습 모델 생성 문장:", result)
    df.loc[1] = curr_percentage    

    print(tabulate(df, headers='keys', tablefmt='fancy_outline'))
    # dqn.model.save('RLModel_V2_7.h5')
    # dqn.model.save('RLModel_V2_17.h5')
    # dqn.model.save('RLModel_V2_40.h5') # 학습 모델 저장
    dqn.model.save('RLModel_V2_49.h5') # 학습 모델 저장

    return jsonify({'success': 'relearning clear'})
######################################################################


######################################################################
# Front(React)와 통신하는 코드                                        #
######################################################################
@app.route('/api/hello', methods=['POST'])
def hello():
    # 1. JSON 데이터 받아오기(POST 방식)_[type='dict']
    request_json = request.get_json()
    print(request_json)

    # 2. dict 데이터 읽어내기
    firstWord = request_json['firstWord']
    secondWord = request_json['secondWord']
    print(firstWord, secondWord)
    
    # 3. 문장 생성
    # result = text_gen_callback.on_epoch_end(firstWord)
    result = text_gen.text_generate_two_words(firstWord, secondWord)
    print("기존 모델 생성 문장:", result)

    # 4. [KIPRIS API]_문장과 연관된 특허 출력
    relationPatentList = text_gen.relationPatent(result)
    
    # 5. 결과 반환
    # return jsonify({'gsentence': result, 'relationPatentList': ["test1", "test2", "test3", "test4", "test5"]})
    return jsonify({'gsentence': result, 'relationPatentList': relationPatentList})
    # return result
#######################################################################

# Spring Boot과 통신하는 코드
#  return text_gen.on_epoch_end()

# 별개의 코드
@app.route('/predict', methods=['POST'])
def predict():
    # POST 요청으로 전송된 데이터 가져오기
    data = request.get_json()

    # 입력 데이터를 Keras 모델이 예측할 수 있는 형태로 변환
    input_data = np.array(data['input'])

    # Keras 모델 실행
    output_data = model.predict(input_data)

    # 결과를 JSON 형태로 반환
    return jsonify({'output': output_data.tolist()})

if __name__ == '__main__':
    app.run()