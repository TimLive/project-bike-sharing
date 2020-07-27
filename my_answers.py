import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes # example 3
        self.hidden_nodes = hidden_nodes # example 2
        self.output_nodes = output_nodes # example 1

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes)) # 3_2

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        self.sigmoid_prime = lambda x : x * (1 - x)
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

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
            
            #X = np.array(X, ndmin=2)
            #targets = np.array(targets, ndmin=2)
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        X = np.array(X, ndmin=2)
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) #  1_3 * 3_2 = 1_2 signals into hidden layer          
        hidden_outputs = self.activation_function(hidden_inputs) # 1_2 ( signals from hidden layer
        #print("hidden_inputs", hidden_inputs.shape, " = ", hidden_inputs)
        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # 1_2 * 2_1 = 1_1 signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        #print("final_outputs", final_outputs.shape, " = ", final_outputs)
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        X = np.array(X, ndmin=2)
        y = np.array(y, ndmin=2)
        hidden_outputs = np.array(hidden_outputs, ndmin=2)
        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
#         print("error", error, ", y, ", y, ", final_outputs", final_outputs)
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(error, self.weights_hidden_to_output.T) # 2_1 * 1_1 = 
#         print("hidden_error", hidden_error)
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        hidden_error_term= hidden_error * hidden_outputs * (1 - hidden_outputs )
#         print("hidden_error_term", hidden_error_term)
        
        output_error_term = error
        
        # Weight step (input to hidden)
        delta_weights_i_h += np.dot(X.T, hidden_error_term) 
#         print("delta_weights_i_h", delta_weights_i_h)
        # Weight step (hidden to output)        
        delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # 
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # 

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        features = np.array(features, ndmin=2)
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.        
#         print("self.input_nodes", self.input_nodes)
#         print("self.weights_input_to_hidden", self.weights_input_to_hidden.shape)
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) #  1_3 * 3_2 = 1_2 signals into hidden layer          
        hidden_outputs = self.activation_function(hidden_inputs) # 1_2 ( signals from hidden layer
        #print("hidden_inputs", hidden_inputs.shape, " = ", hidden_inputs)
        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.4
hidden_nodes = 8
output_nodes = 1