from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
# import numpy as np
from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from math import fsum
import os.path

# implement 3-way classification
# POLARITIES = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
MODEL_FILE = 'trained_models/polarity_model.pkl'
FEAT_VECTOR_FILE = 'trained_models/polarity_feat_vector.pkl'

def extract_features(trainingData):
  """Extract and tag features on the training data
  Assumes tokenization and pos tagging has been done"""
  feature_list = []
  for data_entry in trainingData:
    entry_feature_dict = defaultdict(int)
    for t in data_entry["lemmas"]:
    #   feature : does text contain word
      entry_feature_dict["has_word (" + t.lower() + ")"] = 1
    #   entry_feature_dict["wc (" + t.lower() + ")"] += 1

    for x in range(len(data_entry["lemmas"])-1):
        entry_feature_dict["has bigram (" + data_entry["lemmas"][x].lower() + " " + data_entry["lemmas"][x+1].lower() + ")"] = 1
    feature_list.append(entry_feature_dict)
  return feature_list

def extract_labels(trainingData):
  """product an array of truth labels for the genres"""
  return [y["truth"]["polarity"] for y in trainingData]

def train_model(x, y):
  """produce a trained classifier model
  x : the feature list
  y : the label list
  """
  feat_vector = DictVectorizer().fit(x)
  X_train = feat_vector.transform(x)
  classifier = LinearSVC().fit(X_train, y)

  # save the trained model to disk
  joblib.dump(classifier, MODEL_FILE)
  # save the fitted feature vector
  joblib.dump(feat_vector, FEAT_VECTOR_FILE)

def test_model(x):
  """test the classifier model
  x : the feature list
  y : the label list
  """

  # check if trained model exists
  if not os.path.isfile(MODEL_FILE) or not os.path.isfile(FEAT_VECTOR_FILE):
    print('Genre model "' + MODEL_FILE +'" not found.')
    print('be sure to train model first!')
    return

  feat_vector = joblib.load(FEAT_VECTOR_FILE)
  X_test = feat_vector.transform(x)

  classifier = joblib.load(MODEL_FILE)

  test_pred = classifier.predict(X_test)
  # acc = numpy.mean([xi==yi for xi in test_pred for yi in y])
  # f1_fp_count = len([xi==yi for xi in test_pred for yi in y])
  # print("Accuracy: " + str(acc))
  # print(test_pred)
  return test_pred
