# tensorflow의 다변인 선형회귀 알고리즘 사용, 실제 기상청 데이터를 이용하여 배추가격 예측 ai 모델 개발
import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv # csv파일 읽는 라이브러리 불러오기

data = read_csv('C:/Users/mnb91/Desktop/data/cabbage prediction ai/after refine/real refined data.csv', sep=',') # 콤마를 기준으로 분리(separating)함
xy = np.array(data, dtype=np.float32)

x_data = xy[:, 1:-1] # 두번째 열부터 뒤에서 두번쨰 열까지 슬라이싱 (-1 인덱스는 제일 마지막 열을 의미함, 1:-1이므로 1 ~ -2)
y_data = xy[:, [-1]] # 제일 마지막 열 슬라이싱 (가격값)

X = tf.placeholder(tf.float32, shape=[None, 4]) # 4개의 변인('평균온도', '최저온도', '최고온도', '강수량')이 들어가는 플레이스 홀더 선언
Y = tf.placeholder(tf.float32, shape=[None, 1]) # 1개의 변인(가격)이 들어가는 플레이스 홀더 선언

W = tf.Variable(tf.random_normal([4, 1]), name='Weight') # 가중치(일차 함수 기울기), tf.random_normal은 0~1까지의 랜덤한 수 하나를 지정함(정규분포의 수 하나를 지정)
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b # 가설 설정, 일차함수, tf.matmul은 ([1, 2], [3, 4]) 와 같은 2*2 2차원 행렬의 곱셈을 가능하게 해줌 **** H(x1, x2, x3, x4) = x1w1 + x2w2 + x3w3 + x4w4 (가중치가 각각의 변인에 곱해짐) 하나의 문법이므로 X와 W 인자 위치 바꾸지 말 것!! ****

cost = tf.reduce_mean(tf.square(hypothesis - Y)) # 비용 함수 설정

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005) # 경사하강 라이브러리 사용, 가중치(하강 점프 크기)는 0.000005로 지정
train = optimizer.minimize(cost) # 비용 함수를 최소화 함 (비용이 적을 수록 결과 값의 정확도 상승)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # init 초기화

for step in range(10000): # 10000번 학습 수행
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
    if step % 500 == 0:
        print("#", step, "손실 비용:", cost_)
        print("- 배추 가격:", hypo_[0])

# 학습된 모델을 저장합니다.
saver = tf.train.Saver() # 모델을 저장하기 위한 객체 선언 saver.save() :저장하기, saver.restore() :불러오기
save_path = saver.save(sess, "./saved.cpkt") #check point
print("학습된 모델을 저장했습니다.")
# 이제 위 소스코드를 실행하면 됩니다. 결과적으로 향후 사용자가 날씨 정보를 입력했을 때 배추 가격을 바로 보여줄 수 있도록 학습 모델이 파일 형태로 저장된 것을 확인할 수 있습니다.


