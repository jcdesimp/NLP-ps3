# NLP-ps3
### Natural Language Processing  - Problem Set 3

# installing dependencies
All dependencies should be bundled with an Anaconda install, if any are missing they can be installed using pip and the requirements.txt file. Run
`pip install -r requirements.txt` to install missing dependencies on your system.

# How to Run
Assuming you are in the root directory of the project, the program can be run from the command line with `python3 main.py`.

### Usage
Help output is as follows

```
$ python3 main.py 
usage: main.py [-h] [--train TRAINING_DATA] [--test TEST_DATA]

Train or test classifiers.

optional arguments:
  -h, --help            show this help message and exit
  --train TRAINING_DATA
                        Train the classifiers with the given training data
                        file.
  --test TEST_DATA      Test the classifiers with the given training data
                        file.
```

# Project Structure
* **main.py** - the main program, run this from the commandline
* **/trained_models** - this directory will contained the serialized "pickled" trained models that get generated in "--train" mode. These are what will be used when using the "--test" mode of the progam.
* **/classifiers** - this directory contains separate code for each of the 3 classifiers
	* **__init__.py** - this file tells python to treat this directory as a moduke, allowing us to import the classifiers from within the `main.py` file.
	* **eventType.py** - contains the code for the multi-class event type classifier.
	* **genre.py** - contains the code for the binary genre classifier.
	* **polarity.py** - contains the code for the 3-way polarity classifier