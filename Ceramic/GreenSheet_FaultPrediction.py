import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import warnings
import seaborn as sns
import tqdm
warnings.filterwarnings(action='ignore')


class Visualization:
    def __init__(self) -> None:
        pass

    def data_relations(self, datas):
        # 양품/불량 레이블 별 공정 X인자와 Y인자 간 상관관계 시각화
        sns.pairplot(datas, diag_kind='kde', hue='binary_failure', palette='bright')
        plt.show()

    def relateXY_visualize(self, X, Y):
        # 훈련 데이터 시각화
        trains = pd.concat([X, Y], axis=1)
        trains.hist(figsize=(10, 10))
        plt.show()
        return trains

class DataSet:
    def __init__(self) -> None:
        data_path = './FourMat/Ceramic/ceramic_original_data.csv'
        data = pd.read_csv(data_path)

        self.__feature_name = ['pbratio', 'binder_con', 'defoamer', 'final_temp', 'viscosity',
                   'coating_speed', 'gap', 'lab_temperature', 'lab_humidity', 'simulation']
        self.__target_name = ['binary_failure']

        self.__dataset = data[self.__feature_name + self.__target_name]
        self.__dataset.head(5)

    def data_all(self):
        return self.__dataset

    def data_XY(self):
        return self.__dataset[self.__feature_name], self.__dataset[self.__target_name]
    
    def get_feature_name(self):
        return self.__feature_name
    
    def get_target_name(self):
        return self.__target_name


class Preprocessing:
    def data_division(self, x, y, test_size=0.2):
        # 분류모델 훈련 데이터와 평가 데이터 분할
        return train_test_split(x, y, test_size=test_size)

class Prediction:
    def __init__(self):
        self.__best_model = None
        self.__best_score = .0
        self.__best_model_number = 0

    def make_model(self, X_train, X_eval, y_train, y_eval, epoch):
        # 분류모델 학습 (iteration: 100)
        pbar = tqdm.tqdm(range(epoch))
        for i in pbar:
            model = RandomForestClassifier(n_estimators=100, oob_score=True)
            model.fit(X_train, y_train)
            score = model.score(X_eval, y_eval)
            pbar.set_description(f"{i+1}번째 모델 생성중 - 현재 최적모델 :  {self.__best_model_number+1} ")
            if self.__best_score < score:
                self.__best_score = score
                self.__best_model = model
                self.__best_model_number = i

        print("<< 훈련종료 >>")
        print('훈련 데이터 정확도: {:.3f}'.format(self.__best_model.score(X_train, y_train)))
        print('평가 데이터 정확도: {:.3f}'.format(self.__best_model.score(X_eval, y_eval)))
        print('테스트 데이터 정확도: {:.3f}'.format(self.__best_model.oob_score_))
        return self.__best_model, self.__best_score

    def model_predict(self, trainss, test_data):
        # 불량 예측결과 테이블 시각화 (Real - binary_failure, Pred - predicted_label)
        preds = self.__best_model.predict(test_data)
        trainss['predicted_label'] = preds
        print(trainss.head(10))

    def tree(self, features):
        # 생성된 의사결정 트리 시각화
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=300)
        dot_data = tree.plot_tree(self.__best_model.estimators_[10],
                                  feature_names=features,
                                  class_names=['P', 'F'],
                                  max_depth=5,
                                  precision=3,
                                  filled=True,
                                  rounded=True)