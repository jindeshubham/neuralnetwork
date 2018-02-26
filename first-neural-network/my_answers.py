import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.activation_function = sigmoid  # Replace 0 with your sigmoid calculation.

        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        # def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        # self.activation_function = sigmoid

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            ### Forward pass ###

            print ("Forward pass")
            print ("X")
            print (X.shape)
            print ("weights input to hidden")
            print (self.weights_input_to_hidden.shape)
            print ("weights hidden to output")
            print (self.weights_hidden_to_output.shape)
            # TODO: Hidden layer - Replace these values with your calculations.
            hidden_inputs = np.dot(X.T, self.weights_input_to_hidden)  # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
            print ("Hidden inputs "+str(hidden_inputs.shape))
            print ("Hidden outputs "+str(hidden_outputs.shape))

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs
            print ("Final outputs"+str(final_outputs.shape))
            ### Backward pass ###
            # Output layer
            output_error = y - final_outputs
            output_error_term = output_error

            # Hidden layer
            hidden_error = np.matmul(self.weights_hidden_to_output, output_error_term)
            hidden_gradient = (hidden_outputs * (1 - hidden_outputs))
            hidden_error_term = hidden_error.T * hidden_gradient
            # Weight step (input to hidden)
            delta_weights_i_h += self.lr * (X.reshape((X.shape[0], 1)) * hidden_error_term)
            # Weight step (hidden to output)
            delta_weights_h_o += self.lr * (hidden_outputs.reshape((hidden_outputs.shape[0], 1)) * output_error_term)

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += delta_weights_h_o  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.matmul(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1000
learning_rate = 0.01
hidden_nodes = 8
output_nodes = 1
