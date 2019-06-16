# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:24:10 2019

@author: 86181
"""
from __future__ import print_function

from flask import Flask, request
import flask
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab


base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

app = Flask(__name__)

'''
class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]
'''
@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return flask.render_template("upload.html")

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    data = dict()
    data = {"Success": False}
    if flask.request.method == "POST":
        if flask.request.form["text1"]:
            text1 = flask.request.form["text1"]
            print(text1)
            # get prediction
           # preds = cnn_model.predict(text1)
           # print(preds)
            data["prediction"] = "你好"
            data["Success"] = True
    return flask.jsonify(data)
 
    
    
#global cnn_model
#cnn_model = CnnModel()
if __name__ == "__main__":
    app.run(debug=False)