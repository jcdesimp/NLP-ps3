#!/usr/bin/env python3

import argparse
import sklearn

from nltk.tokenize import word_tokenize
from nltk import pos_tag

from classifiers import genre, polarity, eventType

def parseDataFile(filename):
  """Parse data from file"""
  print("parsing data from file...")
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
  """Tokenize text and add "tokens" property to parsed data"""
  print("tokenizing...")
  for d in parsedData:
    d["tokens"] = word_tokenize(d["text"])

def tagPOS(parsedData):
  """Generate POS tags for tokens and set "pos_tags" property on parsed data
  assumes tokenzation has been done and as on the 'tokens' property of the parsedData entries"""
  print("tagging POSs...")
  for d in parsedData:
    d["pos_tags"] = pos_tag(d["tokens"])

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
    # attach POS tags
    tagPOS(parsedTrainingData)
    # print(parsedTrainingData)
    
  elif args.test:
    print("test: not implemented")
  else:
    parser.print_help()

# If executing this file, run main function
if __name__ == "__main__":
  main()
