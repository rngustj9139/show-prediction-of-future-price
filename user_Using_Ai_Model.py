# 저장된 학습 모델을 불러와서 사용하여 배추 가격 예측하기
import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

init = tf.global_variables_initializer()
saver = tf.train.Saver() # 저장된 모델을 불러오는 객체를 선언합니다.

# 사용자로부터 기상 정보 입력 받음
avg_temp = float(input('평균 온도:')) 
min_temp = float(input('최저 온도:'))
max_temp = float(input('최고 온도:'))
rain_fall = float(input('강수량:'))

with tf.Session() as sess:
    sess.run(init)

    # 저장된 학습 모델을 파일로부터 불러옵니다. saver.restore() :불러오기
    save_path = './saved.cpkt'
    saver.restore(sess, save_path)

    data = ((avg_temp, min_temp, max_temp, rain_fall),) # 하나로 묶기 위해 튜플사용 => 튜플로 한개만 표현할시 꼭 마지막에 콤마가 있어야함 ex) (변수),
    arr = np.array(data)

    x_data = arr[0:4] # 변인 4개

    dict = sess.run(hypothesis, feed_dict={X:x_data})
    print(dict[0])