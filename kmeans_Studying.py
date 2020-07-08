from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sb #데이터 시각화 관련 라이브러리
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['x', 'y'])

df.loc[0] = [5, 8] #첫번쨰 행에 X:5, Y:8 데이터 삽입
df.loc[1] = [1, 12]
df.loc[2] = [8, 15]
df.loc[3] = [15, 4]
df.loc[4] = [11, 2]
df.loc[5] = [6, 9]
df.loc[6] = [16, 18]
df.loc[7] = [4, 4]
df.loc[8] = [8, 6]
df.loc[9] = [1, 14]
df.loc[10] = [16, 19]
df.loc[11] = [8, 16]
df.loc[12] = [1, 5]
df.loc[13] = [9, 1]
df.loc[14] = [3, 7]
df.loc[15] = [10, 11]
df.loc[16] = [3, 5]
df.loc[17] = [9, 12]
df.loc[18] = [18, 4]
df.loc[19] = [7, 2]
df.loc[20] = [8, 8]
df.loc[21] = [17, 5]
df.loc[22] = [12, 15]
df.loc[23] = [2, 18]
df.loc[24] = [2, 11]
df.loc[25] = [9, 5]
df.loc[26] = [5, 14]
df.loc[27] = [12, 8]
df.loc[28] = [6, 5]
df.loc[29] = [4, 11]
df.loc[30] = [15, 2]

sb.lmplot('x', 'y', data=df, fit_reg=False, scatter_kws={'s':100}) # fit_reg=False (선형회귀 선을 안보이게 처리) scatter_kws (그래프 위 점의 사이즈를 100으로 설정)
plt.title('K-means Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.show() # sb.lmplot 명령어 만으로는 pycharm에서 non show 되기 때문에 plt.show() 명령어를 사용하여 결과 값을 show함

d = df.values # pandas 데이터를 numpy 행렬로 변환
kmeans = KMeans(n_clusters=4).fit(d) # 클러스터 개개수4개
print(kmeans.cluster_centers_) # K-means++ 알고리즘을 사용하여 자동으로 중심값을 지정
print(kmeans.labels_) # 31개의 데이터가 어떤 클러스터에 속하는지 출력 (총4개의 클러스터 존재)

df['cluster'] = kmeans.labels_ # '데이터프레임에 cluster'라는 이름의 열을 추가시키고 아래 값들은 kmeans.labels_을 insert
print(df)

sb.lmplot('x', 'y', data=df, fit_reg=False, scatter_kws={'s':100}, hue='cluster') # hue='cluster' 는 클러스터의 종류를 기준으로 색상을 바뀌게함
plt.title('K-means Example two')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



