import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# classificatore bayesiano basato su probabilità ed è lineare
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pieroit/corso_ml_python_youtube_pollo/master/movie_review.csv', sep=',')

print(df.head())

X = df['text']
y = df['tag']

# considero anche una sequenza di due parole
vect = CountVectorizer(ngram_range=(1, 2))
X = vect.fit_transform(X)

# abbiamo aggiunto ngram in countvect per migliorare le accuratezze comunque ottime
# ma ha migliorato solo train e non test, questo significa che inizia a imparare a memoria le coppie di parole che
# distinguono il sentiment (cioè memorizza quale coppia forma le positive e quali le negative
# prendendo solo una parte di dati nel train, quindi performa meglio ma sbagliando) nel training set,
# allora si prova a dare meno dati al train ma risultato uguale
# NOTA: questo lo noto dal fatto che i risultati delle accurancy sono troppo lontani il train aumenta a 95% mentre
# il test rimane a 70 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

model = BernoulliNB()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'Train acc. {acc_train}, test acc. {acc_test}')

# per vedere tutte le parole che contiene un testo, non ordine
# print(vect.inverse_transform(X[:2]))





