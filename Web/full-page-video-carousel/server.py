from flask import Flask, render_template, request
import datetime
import tensorflow as tf
import numpy as np

app = Flask(__name__)

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='Weight') # key weight not found in cpkt 오류 발생시 name='weight' 소문자, 대문자 확인하기
b = tf.Variable(tf.random_normal([1]), name='bias')

# tf.matmul() -> 인자의 곱이 아닌 행렬의 곱 사용
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러오는 객체를 선언합니다.
saver = tf.train.Saver()

model = tf.global_variables_initializer()
sess = tf.Session()
sess.run(model)

# 저장된 모델을 불러옵니다. 세션에 적용합니다.
save_path = "C:/Users/mnb91/PycharmProjects/baechuprice/Web/full-page-video-carousel/model/saved.cpkt"
saver.restore(sess, save_path)

@app.route('/', methods=['GET', 'POST']) # POST 방식만 쓰면 오류발생 가능성 존재, GET방식도 함께 사용
def index():
    if request.method == 'GET': # 사용자의 요청 방식이 GET 일경우
        return render_template('index.html')
    if request.method == 'POST': # 사용자의 요청 방식이 POST 일경우
        # 파라미터를 전달 받습니다.
        avg_temp = float(request.form['avg_temp']) # 사용자가 POST 방식으로 요청한 데이터는 request.form['변수명'] 으로 접근 가능
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rain_fall = float(request.form['rain_fall'])

        price = 0 # 배추 가격 변수를 선언(초기화)합니다.

        data = ((avg_temp, min_temp, max_temp, rain_fall),) # 입력 데이터를 numpy 배열로 한번에 변환하기위해 tuple 자료형으로 변환합니다.
        arr = np.array(data, dtype=np.float32) # 입력된 파라미터들을 numpy배열 형태로 준비합니다.

        x_data = arr[0:4]
        dict = sess.run(hypothesis, feed_dict={X:x_data}) # 입력값을 토대로 결과 값을 찾아냅니다.

        price = dict[0] # 배추 가격 결과를 저장합니다.

        return render_template('index.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)