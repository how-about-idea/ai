import tensorflow as tf
import numpy as np
import random

class DQN:
    def __init__(self, model, orgModel, word_to_index, maxlen=30,learning_rate=0.001):
        self.maxlen = maxlen
#         self.originalModel = originalModel
#         self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.word_to_index = word_to_index
        self.memory = []
        self.model = model
        self.orgModel = orgModel
        self.build_model()

    def build_model(self):
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
    def replay(self):
        
        random.shuffle(self.memory)
        
        # Create the batch using the selected indices
        states = np.array([transition[0] for transition in self.memory])
        indexs = np.array([transition[1] for transition in self.memory])
        targets = np.array([transition[2] for transition in self.memory])
        
        self.memory = []

        self.model.fit([states, indexs], targets, epochs=25, verbose=1)
        # self.model.fit([states, indexs], targets, epochs=25 verbose=1)
#         self.model.save('RLModel.h5')
        
    def train(self, text, reward):
        tokens_index = [self.word_to_index.get(_, 1) for _ in text]
        tokens_index.append(0)

        state = np.zeros(self.maxlen)
        
        for index in range(len(tokens_index) - 1):
            state[index] = tokens_index[index]
            action = tokens_index[index + 1]
            next_state = state.copy()
            
            state = np.array([state])
            targets = self.orgModel.predict(state)[0][0][index]
            targets[action] += reward
            state = state.reshape(-1,)
            targets = targets.reshape(-1,)
            
            self.memory.append((state, index, targets))
            
#             pre = self.model.predict(state)
#             pre = np.array(pre)
#             targets = np.array(targets)
            
            state = next_state.copy()
        self.replay()
            
#             pre = self.model.predict(state)
#             pre = np.array(pre)
#             targets = np.array(targets)
            
#         if len(self.memory) >= batch_size:
#             del self.memory[:batch_size]
        # self.replay(batch_size)
