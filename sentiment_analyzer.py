from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

train = [
         ("i am happy","pos"),
         ("i love this music","pos"),
         ("this is my best friend","pos"),
         ("this is an awesome day","pos"),
         ("i am sad","neg"),
         ("this is an amazing movie","pos"),
         ("i hate this movie","neg"),
         ("i am tired of this stuff","neg"),
         ("i cant deal with this","neg"),
        ]

train_features = []
train_labels = []

for data in train:
        train_features.append(data[0])
        train_labels.append(data[1])
print(train_features)
print(train_labels)

"""
['i am happy', 'i love this music', 'this is my best friend', 'this is an awesome day', 'i am sad', 'this is an amazing movie', 'i hate this movie', 'i am tired of this stuff', 'i cant deal with this']
['pos', 'pos', 'pos', 'pos', 'neg', 'pos', 'neg', 'neg', 'neg']
"""

k = ["balu love mougli","balu love bageera"]

count_vect = CountVectorizer()
sample = count_vect.fit_transform(k)
print(sample.toarray())

"""
[[0 1 1 1]
 [1 1 1 0]]
"""
k = ["balu love pickachu"]
k = count_vect.transform(k)
print(k.toarray())

"""
[[0 1 1 0]]
"""

x_train = count_vect.fit_transform(train_features)

print(x_train.toarray())

"""
[[1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0]
 [0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0]
 [0 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0]
 [0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1]]
"""
y_train = []
for data in train_labels:
        if data == "pos":
                y_train.append(0)
        elif data == "neg":
                y_train.append(1)
print(y_train)

"""
[0, 0, 0, 0, 1, 0, 1, 1, 1]
"""

classifier = GaussianNB()
classifier.fit(x_train.toarray(),y_train)
k = count_vect.transform(["i am crazy"])

output = classifier.predict(k.toarray())
#print(output)

"""
[0]
"""

"""
[1]
"""
if output[0] == 0:
        print("positve")
else:
        print("negative")
"""
negative
"""