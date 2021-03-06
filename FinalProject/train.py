# coding: utf-8
# 2020/인공지능/final/학번/이름
import sys, os
import argparse
import time
sys.path.append(os.pardir)

import numpy as np
from AReM import *
from model import *


class Trainer:
    """
    ex) 200개의 훈련데이터셋, 배치사이즈=5, 에폭=1000 일 경우 :
    40개의 배치(배치당 5개 데이터)를 에폭 갯수 만큼 업데이트 하는것.=
    (200 / 5) * 1000 = 40,000번 업데이트.

    ----------
    network : 네트워크
    x_train : 트레인 데이터
    t_train : 트레인 데이터에 대한 라벨
    x_test : 발리데이션 데이터
    t_test : 발리데이션 데이터에 대한 라벨
    epochs : 에폭 수
    mini_batch_size : 미니배치 사이즈
    learning_rate : 학습률
    verbose : 출력여부

    ----------
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=1000, mini_batch_size=100,
                 learning_rate=0.0003, verbose=True, layers = [6, 64, 64, 64, 64, 6]):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = int(epochs)
        self.batch_size = int(mini_batch_size)
        self.lr = learning_rate
        self.verbose = verbose
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = int(max(self.train_size / self.batch_size, 1))
        self.max_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
        self.layers = layers # [6, 64, 64, 64, 64, 6]
        self.layer_size = len(layers) # 6
        self.test_acc = None # Test_acc 확인위해 설정
        
    def train_step(self):
        # 렌덤 트레인 배치 생성
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 네트워크 업데이트, Train Update에서는 dropOut = True로
        self.network.update(x_batch, t_batch, True) 
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            train_acc, _ = self.accuracy(self.x_train, self.t_train)
            test_acc, _ = self.accuracy(self.x_test, self.t_test)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            if self.verbose: 
                print("=== epoch:", str(round(self.current_epoch, 3)), ", iteration:", str(round(self.current_iter, 3)),
                ", train acc:" + str(round(train_acc, 3)), ", test acc:" + str(round(test_acc, 3)), ", train loss:" + str(round(loss, 3)) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()
        self.test_acc, inference_time = self.accuracy(self.x_test, self.t_test)
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(self.test_acc) + ", inference_time:" + str(inference_time))
            print('[size = {}][epoch = {}][batch = {}][lr = {}][layer = {}]'\
                        .format(self.layer_size, self.epochs, self.batch_size, self.lr, self.layers))
            print("")

    def accuracy(self, x, t):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0
        start_time = time.time()
        
        y = self.network.predict(x)
        y = np.argmax(y, axis=1)
        acc += np.sum(y == t)

        inference_time = (time.time() - start_time) / x.shape[0]

        return acc / x.shape[0], inference_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py --help 로 설명을 보시면 됩니다."
                                                 "사용예)python train.py --sf=myparam --epochs=10")
    parser.add_argument("--sf", required=False, default="params.pkl", help="save_file_name")
    parser.add_argument("--epochs", required=False, default=1000, help="epochs : default=500")
    parser.add_argument("--mini_batch_size", required=False, default=100, help="mini_batch_size : default=100")
    parser.add_argument("--learning_rate", required=False, default=0.0003, help="learning_rate : default=0.0001")
    args = parser.parse_args()

    # 데이터셋 탑재
    (x_train, t_train), (x_test, t_test) = load_AReM(one_hot_label=False)
    
    mean = np.mean(x_train, axis = 0)
    std = np.std(x_train, axis=0)
    
# hyperparameter
# 레이어 층과 unit수 설정 input = 6, hidden = [64, 64, 64, 64], output = 6
    layer_unit = [6, 64, 64, 64, 64, 6]
    # 모델 초기화 featureScaling(mean, std), dropout=True, layer전달
    network = Model(mean, std, True, layer_unit)

    # 트레이너 초기화
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=args.epochs, mini_batch_size=args.mini_batch_size,
                      learning_rate=args.learning_rate, verbose=True, 
                      layers = layer_unit)

    # 트레이너를 사용해 모델 학습
    trainer.train()
    # 파라미터 보관
    network.save_params(args.sf)
    print("Network params Saved ")

