import glob
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app)


@app.route("/")
def get_data():
    return "Group 16 Service Hit!"


@app.route("/predict", methods=['POST'])
def predict():
    req_data = request.json

    model_1 = pickle.load(open('model1.pkl', 'rb'))
    model_2 = pickle.load(open('model2.pkl', 'rb'))
    model_3 = pickle.load(open('model3.pkl', 'rb'))
    model_4 = pickle.load(open('model4.pkl', 'rb'))

    res = []
    for video_index in range(0, len(req_data)):

        labels = dict()
        df = pd.DataFrame(columns=column_names)
        for i in range(0, len(req_data[video_index])):
            df = populate_data(df, req_data[video_index][i])

        labels[1] = predict_unseen_data(df, model_1)
        labels[2] = predict_unseen_data(df, model_2)
        labels[3] = predict_unseen_data(df, model_3)
        labels[4] = predict_unseen_data(df, model_4)
        res.append(labels)

    return json.dumps(res)


# Machine Learning Script

column_names = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
                'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
                'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
                'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
                'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
                'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
                'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
                'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
                'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']

char_labels = dict()
char_labels['buy'] = 0
char_labels['communicate'] = 1
char_labels['fun'] = 2
char_labels['hope'] = 3
char_labels['mother'] = 4
char_labels['really'] = 5

num_labels = dict()
num_labels[0] = 'buy'
num_labels[1] = 'communicate'
num_labels[2] = 'fun'
num_labels[3] = 'hope'
num_labels[4] = 'mother'
num_labels[5] = 'really'


def populate_data(df, json_data):
    row = dict()
    row['score_overall'] = json_data['score']
    temp = json_data['keypoints']
    for i in range(0, len(temp)):
        part = temp[i]['part']
        row[part + '_score'] = temp[i]['score']
        row[part + '_x'] = temp[i]['position']['x']
        row[part + '_y'] = temp[i]['position']['y']
    df = df.append(row, ignore_index=True)

    return df


def read_data(folder_name):
    folders = glob.glob(folder_name + '/*')
    frame_data = pd.DataFrame()
    for folder in folders:
        files = glob.glob(folder + '/*.csv')
        for file in files:
            temp = pd.read_csv(file)
            temp = temp.iloc[:, :35]
            key = folder[folder.index('/') + 1:]
            class_label = pd.DataFrame(np.full((temp.shape[0], 1), char_labels[key]),
                                       columns=['class'])
            temp = pd.concat([temp, class_label], axis=1)
            frame_data = pd.concat([frame_data, temp])

    frame_data = frame_data.iloc[:, 1:].values
    return frame_data


def build_model(frame_data):
    x = frame_data[:, :-1]
    y = frame_data[:, 34]

    # Model building
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=10)
    rf.fit(x, y)

    lr = LogisticRegression(random_state=10)
    lr.fit(x, y)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x, y)

    nb = GaussianNB()
    nb.fit(x, y)

    return rf, lr, knn, nb


def predict_unseen_data(unseen_data, model):
    x_test = unseen_data.iloc[:, :34].values

    y_prediction = model.predict(x_test)

    class_freq = dict()
    for i in range(0, len(y_prediction)):
        if y_prediction[i] in class_freq:
            class_freq[y_prediction[i]] += 1
        else:
            class_freq[y_prediction[i]] = 1

    max_count = 0
    gesture = -1
    for key in class_freq:
        if class_freq[key] > max_count:
            max_count = class_freq[key]
            gesture = key

    return num_labels[int(gesture)]


# def read_test_data(folder_name):
#     frame_data = pd.DataFrame()
#
#     files = glob.glob(folder_name + '/*.csv')
#     for file in files:
#         temp = pd.read_csv(file)
#         frame_data = pd.concat([frame_data, temp])
#
#     frame_data = frame_data.iloc[:, 1:35]
#     return frame_data


if __name__ == '__main__':
    # data = read_data('CSV')
    # rf, lr, knn, nb = build_model(data)
    # print('Model Ready!')
    #
    # pickle.dump(rf, open('model1.pkl', 'wb'))
    # pickle.dump(lr, open('model2.pkl', 'wb'))
    # pickle.dump(knn, open('model3.pkl', 'wb'))
    # pickle.dump(nb, open('model4.pkl', 'wb'))

    # model_1 = pickle.load(open('model1.pkl', 'rb'))
    # model_2 = pickle.load(open('model2.pkl', 'rb'))
    # model_3 = pickle.load(open('model3.pkl', 'rb'))
    # model_4 = pickle.load(open('model4.pkl', 'rb'))
    #
    # data = read_test_data('test/really')
    #
    # for model in [model_1, model_2, model_3, model_4]:
    #     print(predict(data, model))

    app.run()

    # for gesture in ['buy', 'communicate', 'fun', 'hope', 'mother', 'really']:
    #     test_data = read_test_data('test/' + gesture)
    #     predict(test_data, rf)
    #     print('True Label - ' + gesture)
    #     print('-------------------------------------------------------------------------------------------------')
