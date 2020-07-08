# 텐서플로우의 선형회귀 알고리즘을 이용한 수익예측 AI
import tensorflow as tf

xData = [1, 2, 3, 4, 5, 6, 7] # 일한 시간
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000] # 수익

W = tf.Variable(tf.random_uniform([1], -100, 100)) # 랜덤한 가중치(기울기)
b = tf.Variable(tf.random_uniform([1], -100, 100))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

H = W * X + b # 일차함수

cost = tf.reduce_mean(tf.square(H - Y)) # 비용함수, 비용이 적을수록 수익예측 정확도가 높아짐

a = tf.Variable(0.01) # 경사하강 점프 가중치(점프의 크기를 결정)

optimizer = tf.train.GradientDescentOptimizer(a) # 경사하강 라이브러리 사용
train = optimizer.minimize(cost) # 비용을 최소화

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(5001): # 5000번 학습 수행
    sess.run(train, feed_dict={X:xData, Y:yData})
    if i % 500 == 0:
        print(i, sess.run(cost, feed_dict={X:xData, Y:yData}), sess.run(W), sess.run(b))
print(sess.run(H, feed_dict={X:[8]})) # 일한 시간이 8시간일때 수익 예측, 출력
