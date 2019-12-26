#%%
import numpy as np

#%% Initialize weights and biases
weights = np.around(np.random.uniform(size = 6), decimals = 2)
biases = np.around(np.random.uniform(size = 3), decimals = 2)

#%% input
x_1 = 0.5
x_2 = 0.85

#%% outputs
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]

a_11 = 1 / (1 + np.exp(-z_11))
a_12 = 1 / (1 + np.exp(-z_12))

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]

a_2 = 1 / (1 + np.exp(-z_2))

#%% Initialization
n = 2 # number of hidden layers
num_hidden_layers = 2  # number of hidden layers
m = [2, 2]  # number of nodes in each hidden layer
num_nodes_output = 1  # number of nodes in the output layer

# loop through each layer and randomly initialize the weights and biases associated with each node
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):

    num_nodes_previous = num_inputs

    network = {}

    for layer in range(num_hidden_layers + 1):

        # determine name of layer
        if layer == num_hidden_layers:
            layer_name = "output"
            num_nodes = num_nodes_output
        else:
            layer_name = f"layer_{layer + 1}"
            num_nodes = num_nodes_hidden[layer]

        # initialize weights and biases associated with each node in the current layer
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = f"node_{node+1}"
            network[layer_name][node_name] = {
                "weights" : np.around(np.random.uniform(size = num_nodes_previous), decimals = 2),
                "bias": np.around(np.random.uniform(size = 1), decimals = 2)
            }

        num_nodes_previous = num_nodes

    return(network)

Net = initialize_network(5, 3, [3, 2, 3], 1)

#%% Compute Weighted Sum at Each Node
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

# Generate 5 inputs that we can feed to Net
np.random.seed(12)
inputs = np.around(np.random.uniform(size = 5), decimals = 2)

# Use compute_weighted_sum function to compute the weighted sum at the first node in the first hidden layer
first_node_weights = Net["layer_1"]["node_1"]["weights"]
first_node_bias = Net["layer_1"]["node_1"]["bias"]

first_node = compute_weighted_sum(inputs, first_node_weights, first_node_bias)

#%% Compute Node Activation
def node_activation(weighted_sum):
    return 1 / (1 + np.exp(-1 * weighted_sum))

first_node_activated = node_activation(first_node)

#%% Forward Propagation

def forward_propagate(network, inputs):
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for layer_node in layer_data:

            node_data = layer_data[layer_node]

            # Compute the weighted sum and the output of each node at teh same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data["weights"], node_data["bias"]))
            layer_outputs.append(np.around(node_output[0], decimals = 4))

        if layer != "output":
            print(f"The output of the nodes in hidden layer number {layer.split('_')[1]} is {layer_outputs}")

        layer_inputs = layer_outputs # set the output of thisr to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions

prediction = forward_propagate(Net, inputs)