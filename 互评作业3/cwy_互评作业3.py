import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

nltk.download('punkt')
nltk.download('stopwords')




def preprocess(raw_text):
    raw_text = raw_text.lower()  # 转化为小写
    words = word_tokenize(raw_text)  # 分词
    words = [word for word in words if word.isalpha()]  # 去除非字母字符
    stop_words = set(stopwords.words('english'))  # 获取停用词表
    words = [word for word in words if word not in stop_words]  # 去除停用词
    clean_text = ' '.join(words)  # 合并成一个新的字符串
    return clean_text

def read_data():
    texts=[]
    for f1 in os.listdir("data/20news"):
        for f2 in os.listdir("data/20news/"+f1):
            fp="data/20news/"+f1+"/"+f2
            try:
                with open(fp, mode="r", encoding="utf-8") as f:
                    text="".join(f.readlines())
                    texts.append(text)
            except:
                pass
    return texts



if __name__=="__main__":
    raw_texts=read_data()
    preprocessed_texts = [preprocess(raw_text) for raw_text in raw_texts]
    print(f"count of texts: {len(preprocessed_texts)}")

    # 向量化
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_texts)

    # 聚类
    k = 20
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    # 分析结果
    silhouette_coefficient = silhouette_score(X, kmeans.labels_)
    ch_index = calinski_harabasz_score(X.toarray(), kmeans.labels_)
    print("Silhouette Coefficient:", silhouette_coefficient)
    print("Calinski-Harabasz Index:", ch_index)

    # 可视化
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X.toarray())
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title('K-means Clustering Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
    plt.savefig("互评作业3/result.png")