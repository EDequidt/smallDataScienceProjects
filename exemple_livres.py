
import sklearn.feature_extraction.text as txt

def comparison_test(texte):    
    count_vectorizer = txt.CountVectorizer(
        binary=True, max_features=20)
    count_vectorizer.fit(texte)
    vectorized = count_vectorizer.transform(texte)
    return vectorized.toarray()
