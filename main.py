#!/usr/bin/env python3

import argparse
import nltk

def processTrainingData(filename):
    parsedData = []
    # parsed data will be a list of
    # dictionaries in the format:
    # 
    # {
    #   "id": "ID",
    #   "text": "The Text..."
    #   "topic": "TOPIC",
    #   "polarity": "POLARITY",
    #   "genre": "GENRE"
    # }
    #
    openFile = open(filename)
    for line in openFile:
        line = line.strip()
        splitLine = line.split('\t')
        parsedLine = {
          "id": splitLine[0],
          "text": splitLine[1],
        }
        print(parsedLine)
        parsedData.append(parsedLine);

def main():
  """Run main program execution."""

  # configure program arguments
  parser = argparse.ArgumentParser(description='Train or test classifiers.')
  parser.add_argument('--train', required=False, metavar='TRAINING_DATA', help='Train the classifiers with the given training data file.')
  parser.add_argument('--test', required=False, metavar='TEST_DATA', help='Test the classifiers with the given training data file.')
  args = parser.parse_args()
  if args.train:
    processTrainingData(args.train)
  elif args.test:
    print("test: not implemented")
  else:
    parser.print_help()

# If executing this file, run main function
if __name__ == "__main__":
  main()