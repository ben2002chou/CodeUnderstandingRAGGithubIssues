import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

stop_words = set(stopwords.words('english'))

def preprocess_text_with_nltk(text):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def sparse_retrieve(post, code):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([post] + code['function'].tolist())
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    code['similarity'] = cosine_sim

    code = code.sort_values(by='similarity', ascending=False)
    code = code.reset_index(drop=True)
    res = code.head(1)

    return res.loc[0, 'function'], (res.loc[0, 'f_path'], res.loc[0, 'start_row'])


if __name__ == "__main__":
    post = """How does the WarpSampler class ensure that multiple processes can safely share data through the result_queue?self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))"""
    code = pd.read_csv("./output/embedded.csv")
    sparse_retrieve(post, code)
    
