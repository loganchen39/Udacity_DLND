import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        # self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
        #                                (self.input_nodes, self.hidden_nodes))
        # self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
        #                                (self.hidden_nodes, self.output_nodes))

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))        
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1.0/(1+np.exp(-x))
        self.activation_prime    = sigmoid_prime
        
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
            #print('In train, X.shape & y.shape')
            #print(X.shape)
            #print(y.shape)
            X = X.reshape((X.shape[0], 1))  # X.reshape((1, X.shape[0])) won't change X's shape, you have to re-assign to X!
            #y = y[0]
            #print(X.shape)
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
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        #print("In forward_pass_train(self, X), X.shape: ")
        #print(X.shape)  # Ligang: testing
        #print("In forward_pass_train(self, X), X.shape: ")
        #print(X.shape)  # Ligang: testing
        
        hidden_inputs = np.matmul(self.weights_input_to_hidden, X)
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs  # activiation function: f(x)=x.
        
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
        dE_dZ2 = -(y-final_outputs)
        dE_dW2 = dE_dZ2 * hidden_outputs.T
        
        dE_dZ1 = dE_dZ2 * self.weights_hidden_to_output.T * self.activation_prime(np.matmul(self.weights_input_to_hidden, X))
        dE_dW1 = np.matmul(dE_dZ1, X.T)
        
        # Weight step (hidden to output)
        delta_weights_h_o += -dE_dW2
        # Weight step (input to hidden)
        delta_weights_i_h += -dE_dW1

        return delta_weights_i_h, delta_weights_h_o

    
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records

        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        ##features.reshape(features.shape[0]), 1)
        #print('In run, features.shape: ')
        #print(features.shape)
        #print('self.weights_input_to_hidden.shape: ')
        #print(self.weights_input_to_hidden.shape)
        hidden_inputs = np.matmul(self.weights_input_to_hidden, features.T)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        #print('hidden_outputs.shape: ')
        #print(hidden_outputs.shape)
        final_inputs = np.matmul(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs.reshape(final_inputs.shape[1])
        #final_outputs = final_inputs
        #print('final_outputs.shape: ')
        #print(final_outputs.shape)
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 8000  # 100
learning_rate = 0.3  # 0.0005  # 0.1
hidden_nodes = 20  # 2
output_nodes = 1
