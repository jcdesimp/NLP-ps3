#!/usr/bin/env python3

import argparse
import sklearn

from nltk.tokenize import word_tokenize

from classifiers import genre, polarity, eventType

def parseDataFile(filename):
    parsedData = []
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

def tokenizeText(parsedData):
  """Tokenize text and add "Tokens" property to parsed data"""
  for d in parsedData:
    d["tokens"] = word_tokenize(d["text"])

def main():
  """Run main program execution."""
  # configure program arguments
  parser = argparse.ArgumentParser(description='Train or test classifiers.')
  parser.add_argument('--train', required=False, metavar='TRAINING_DATA', help='Train the classifiers with the given training data file.')
  parser.add_argument('--test', required=False, metavar='TEST_DATA', help='Test the classifiers with the given training data file.')
  args = parser.parse_args()
  if args.train:
    parsedTrainingData = parseDataFile(args.train)
    # attach tokens
    tokenizeText(parsedTrainingData)
    # print(parsedTrainingData)
    
  elif args.test:
    print("test: not implemented")
  else:
    parser.print_help()

# If executing this file, run main function
if __name__ == "__main__":
  main()
