import nltk
from nltk.tokenize import word_tokenize # or use some other tokenizer

train = [  ('dont be a cry baby. ','neg')]

all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]

#print (t)
classifier = nltk.NaiveBayesClassifier.train(t)
#classifier.show_most_informative_features()
text= input("Enter a sentence: ")
text_for_test={word.lower(): (word in word_tokenize(text.lower())) for word in all_words}
print(classifier.classify(text_for_test))
