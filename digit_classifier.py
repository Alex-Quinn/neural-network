import sys

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

NUMBER_OF_CLASSES = 10

def create_data_set(file_name):
    raw_data = open(file_name).readlines()

    data_set = ClassificationDataSet(64, nb_classes=NUMBER_OF_CLASSES, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for line in raw_data:
        # Get raw line into a list of integers
        line = map(lambda x: int(x), line.strip().split(','))
        data_set.appendLinked(line[:-1], line[-1])
    return data_set

# Create data sets
train_data = create_data_set(sys.argv[1])
test_data = create_data_set(sys.argv[2])
