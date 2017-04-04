from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import os.path

# implement binary classification
GENRES = ['GENRE_A', 'GENRE_B']
MODEL_FILE = 'trained_models/genre_model.pkl'
FEAT_VECTOR_FILE = 'trained_models/genre_feat_vector.pkl'

def extract_features(trainingData):
  """Extract and tag features on the training data
  Assumes tokenization and pos tagging has been done"""
  feature_list = []
  for data_entry in trainingData:
    entry_feature_dict = defaultdict(lambda : 0)
    # entry_feature_dict = dict()
    # entry_feature_dict["token_count"] = len(data_entry["tokens"])
    for t in data_entry["tokens"]:
      # feature : does text contain word
      entry_feature_dict["has_word (" + t.lower() + ")"] = 1
      entry_feature_dict["wc (" + t.lower() + ")"] += 1

    feature_list.append(entry_feature_dict)
  return feature_list

def extract_labels(trainingData):
  """product an array of truth labels for the genres"""
  return [y["truth"]["genre"] for y in trainingData]

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

def test_model(x, y):
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




