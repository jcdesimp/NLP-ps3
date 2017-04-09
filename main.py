#!/usr/bin/env python3

import argparse
import sklearn

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy

from classifiers import genre, polarity, event_type

def parseDataFile(filename):
  """Parse data from file"""
  # print("parsing data from file...")
  # parsed data will be a list of
  # dictionaries in the format:
  #
  # {
  #   "id": "ID",
  #   "text": "The Text...",
  #   "truth": {
  #     "topic": "TOPIC",
  #     "polarity": "POLARITY",
  #     "genre": "GENRE"
  #   }
  # }
  #
  parsedData = []
  openFile = open(filename)
  for line in openFile:
    line = line.strip()
    splitLine = line.split('\t')
    parsedLine = {
      "id": splitLine[0],
      "text": splitLine[1],
      "truth": {
        "topic": splitLine[3],
        "polarity": splitLine[2],
        "genre": splitLine[4]
      }
    }
    parsedData.append(parsedLine);
  return parsedData;

def l_pos(pos):
    if pos[0].lower() == 'n' or pos[0].lower() == 'v' or pos[0].lower() == 'r':
          return pos[0].lower()
    else:
        return ''

def lemmatize_with_pos(wnl, words):
    lems = []
    for t in words:
        p = l_pos(t[1])
        if p == '':
            lems.append(wnl.lemmatize(t[0]))
        else:
            lems.append(wnl.lemmatize(t[0], p))
    return lems

def tokenizeText(parsedData):
  """Tokenize text and add "tokens" property to parsed data"""
  # print("tokenizing...")
  for d in parsedData:
    d["tokens"] = word_tokenize(d["text"])

def lemmatizeText(parsedData):
  """Lemmatize text and add "lemmas" property to parsed data"""
  wnl = WordNetLemmatizer()
  for d in parsedData:
    pos = pos_tag(word_tokenize(d["text"].lower()))
    lems = lemmatize_with_pos(wnl, pos)
    # d["tokens"] = word_tokenize(d["text"])
    d["lemmas"] = lems

def tagPOS(parsedData):
  """Generate POS tags for tokens and set "pos_tags" property on parsed data
  assumes tokenzation has been done and as on the 'tokens' property of the parsedData entries"""
  # print("tagging POSs...")
  for d in parsedData:
    d["pos_tags"] = pos_tag(d["tokens"])

def tagNER(parsedData):
  """Generate NER tags for tokens and set "ner_tags" property on parsed data
  assumes pos tagging has been done and as on the 'pos_tags' property of the parsedData entries"""
  for d in parsedData:
    d["ner_tags"] = ne_chunk(d["pos_tags"], binary=True)

def preprocess(raw_data):
  """Perform all preprocessing tasks"""
  parsedData = parseDataFile(raw_data)
  # attach tokens
  tokenizeText(parsedData)
  # attach POS tags
  tagPOS(parsedData)
  # attach lemmas
  lemmatizeText(parsedData)
  #attach NER tags
  tagNER(parsedData)

  return parsedData


def main():
  """Run main program execution."""
  # configure program arguments
  parser = argparse.ArgumentParser(description='Train or test classifiers.')
  parser.add_argument('--train', required=False, metavar='TRAINING_DATA', help='Train the classifiers with the given training data file.')
  parser.add_argument('--test', required=False, metavar='TEST_DATA', help='Test the classifiers with the given training data file. Prints out raw prediction data by default')
  parser.add_argument('-m', required=False, action='store_true', help='Print basic test metrics instead of raw predictions.')
  args = parser.parse_args()

  if args.train:
    parsedTrainingData = preprocess(args.train)

    # extract features for genre classifier
    genre_features = genre.extract_features(parsedTrainingData)
    genre_labels = genre.extract_labels(parsedTrainingData)
    genre.train_model(genre_features, genre_labels)

    # extract features for topic classifier
    filterdNotNoneData = [x for x in parsedTrainingData if x["truth"]["topic"] != "NONE"]
    topic_features = event_type.extract_features(filterdNotNoneData)
    topic_labels = event_type.extract_labels(filterdNotNoneData)
    event_type.train_model(topic_features, topic_labels)

    # extract features for genre classifier
    polarity_features = polarity.extract_features(parsedTrainingData)
    polarity_labels = polarity.extract_labels(parsedTrainingData)
    polarity.train_model(polarity_features, polarity_labels)

  elif args.test:

    parsedTestData = preprocess(args.test)
    # extract features for genre classifier
    genre_features = genre.extract_features(parsedTestData)
    genre_labels = genre.extract_labels(parsedTestData)
    genre_predictions = genre.test_model(genre_features)

    # extract features for topic classifier
    filterdNotNoneData = [x for x in parsedTestData if x["truth"]["topic"] != "NONE"]
    topic_features = event_type.extract_features(filterdNotNoneData)
    topic_labels = event_type.extract_labels(filterdNotNoneData)
    topic_predictions = event_type.test_model(topic_features)

    polarity_features = polarity.extract_features(parsedTestData)
    polarity_labels = polarity.extract_labels(parsedTestData)
    polarity_predictions = polarity.test_model(polarity_features)

    # todo other predictions
    predictions = []
    if args.m:
      genre_accuracy = numpy.mean([genre_predictions[i]==genre_labels[i] for i in range(len(genre_predictions))])
      topic_accuracy = numpy.mean([topic_predictions[i]==topic_labels[i] for i in range(len(topic_predictions))])
      polarity_accuracy = numpy.mean([polarity_predictions[i]==polarity_labels[i] for i in range(len(polarity_predictions))])
      print("Genre Accuracy: " + str(genre_accuracy))
      print("Topic Accuracy: " + str(topic_accuracy))
      print("Polarity Accuracy: " + str(polarity_accuracy))
    else: # print raw predictions
      for i, e in enumerate(parsedTestData):
        single_result = []
        single_result.append(e["id"])
        single_result.append(e["text"])
        single_result.append(polarity_predictions[i])
        single_result.append(topic_predictions[i])
        single_result.append(genre_predictions[i])

        predictions.append("\t".join(single_result) + "\t")
      print("\n".join(predictions))

  else:
    parser.print_help()

# If executing this file, run main function
if __name__ == "__main__":
  main()
