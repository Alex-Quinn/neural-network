import sys

from pybrain.datasets import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer

def create_data_set(file_name):
    raw_data = open(file_name).readlines()

    data_set = ClassificationDataSet(64, nb_classes=10, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for line in raw_data:
        # Get raw line into a list of integers
        line = map(lambda x: int(x), line.strip().split(','))
        data_set.appendLinked(line[:-1], line[-1])
    return data_set

def build_network(data_set):
    network = FeedForwardNetwork()
    
    # No hidden
    input_layer = LinearLayer(data_set.indim)
    output_layer = SigmoidLayer(data_set.outdim)
    network.addInputModule(input_layer)
    network.addOutputModule(output_layer)
    in_to_out = FullConnection(input_layer, output_layer)
    network.addConnection(in_to_out)
    
    # One hidden
    # input_layer = LinearLayer(data_set.indim)
    # hidden_layer = SigmoidLayer(64)
    # output_layer = LinearLayer(data_set.outdim)
    # network.addInputModule(input_layer)
    # network.addModule(hidden_layer)
    # network.addOutputModule(output_layer)
    # in_to_hid = FullConnection(input_layer, hidden_layer)
    # hid_to_out = FullConnection(hidden_layer, output_layer)
    # network.addConnection(in_to_hid)
    # network.addConnection(hid_to_out)

    # Custom
    # input_layer = LinearLayer(data_set.indim)
    # hidden_layer_1_1 = SigmoidLayer(16)
    # hidden_layer_1_2 = SigmoidLayer(16)
    # hidden_layer_2_1 = SigmoidLayer(4)
    # hidden_layer_2_2 = SigmoidLayer(4)
    # hidden_layer_2_3 = SigmoidLayer(4)
    # hidden_layer_2_4 = SigmoidLayer(4)
    # output_layer = LinearLayer(data_set.outdim)

    # network.addInputModule(input_layer)
    # network.addModule(hidden_layer_1_1)
    # network.addModule(hidden_layer_1_2)
    # network.addModule(hidden_layer_2_1)
    # network.addModule(hidden_layer_2_2)
    # network.addModule(hidden_layer_2_3)
    # network.addModule(hidden_layer_2_4)
    # network.addOutputModule(output_layer)

    # in_to_hid_1_1 = FullConnection(input_layer, hidden_layer_1_1)
    # in_to_hid_1_2 = FullConnection(input_layer, hidden_layer_1_2)
    # hid_1_1_to_hid_2_1 = FullConnection(hidden_layer_1_1, hidden_layer_2_1)
    # hid_1_1_to_hid_2_2 = FullConnection(hidden_layer_1_1, hidden_layer_2_2)
    # hid_1_1_to_hid_2_3 = FullConnection(hidden_layer_1_1, hidden_layer_2_3)
    # hid_1_1_to_hid_2_4 = FullConnection(hidden_layer_1_1, hidden_layer_2_4)
    # hid_1_2_to_hid_2_1 = FullConnection(hidden_layer_1_2, hidden_layer_2_1)
    # hid_1_2_to_hid_2_2 = FullConnection(hidden_layer_1_2, hidden_layer_2_2)
    # hid_1_2_to_hid_2_3 = FullConnection(hidden_layer_1_2, hidden_layer_2_3)
    # hid_1_2_to_hid_2_4 = FullConnection(hidden_layer_1_2, hidden_layer_2_4)
    # hid_2_1_to_out = FullConnection(hidden_layer_2_1, output_layer)
    # hid_2_2_to_out = FullConnection(hidden_layer_2_2, output_layer)
    # hid_2_3_to_out = FullConnection(hidden_layer_2_3, output_layer)
    # hid_2_4_to_out = FullConnection(hidden_layer_2_4, output_layer)
    # network.addConnection(in_to_hid_1_1)
    # network.addConnection(in_to_hid_1_2)
    # network.addConnection(hid_1_1_to_hid_2_1)
    # network.addConnection(hid_1_1_to_hid_2_2)
    # network.addConnection(hid_1_1_to_hid_2_3)
    # network.addConnection(hid_1_1_to_hid_2_4)
    # network.addConnection(hid_1_2_to_hid_2_1)
    # network.addConnection(hid_1_2_to_hid_2_2)
    # network.addConnection(hid_1_2_to_hid_2_3)
    # network.addConnection(hid_1_2_to_hid_2_4)
    # network.addConnection(hid_2_1_to_out)
    # network.addConnection(hid_2_2_to_out)
    # network.addConnection(hid_2_3_to_out)
    # network.addConnection(hid_2_4_to_out)

    network.sortModules()

    return network

# Create data sets
train_data = create_data_set(sys.argv[1])
test_data = create_data_set(sys.argv[2])
train_data._convertToOneOfMany()
test_data._convertToOneOfMany()

# Create network
train_net = build_network(train_data)

print "*******Before training********"
sq_err = []
for data in test_data:
    input_entry = data[0]
    output_entry = data[1]
    pred_entry = train_net.activate(input_entry)
    sq_err.append((pred_entry[0] - output_entry[0])**2)

    # print 'Actual:', output_entry, 'Predicted', pred_entry

print "RMSE: %.2f" % (sum(sq_err) / len(sq_err))

# Train network
trainer = BackpropTrainer(train_net, train_data)
trainer.trainUntilConvergence(maxEpochs=100)

print "*******After training********"
sq_err = []
for data in test_data:
    input_entry = data[0]
    output_entry = data[1]
    pred_entry = train_net.activate(input_entry)
    # print 'Actual:', output_entry, 'Predicted', pred_entry

    sq_err.append((pred_entry[0] - output_entry[0])**2)

print "RMSE: %.2f" % (sum(sq_err) / len(sq_err))
    