from numpy import *
class NeuralNet():
    def __init__(self):
        #Variable initialization
        #epoch=8000 #Setting training iterations
        self.lr=0.1 #Setting learning rate
        self.inputlayer_neurons = 3 #number of features in data set
        self.hiddenlayer_neurons = 3 #number of hidden layers neurons
        self.output_neurons = 1 #number of neurons at output layer
        #weight and bias initialization
        random.seed(1)
        self.wh=random.uniform(size=(self.inputlayer_neurons,self.hiddenlayer_neurons))
        self.bh=random.uniform(size=(1,self.hiddenlayer_neurons))
        self.wout=random.uniform(size=(self.hiddenlayer_neurons,self.output_neurons))
        self.bout=random.uniform(size=(1,self.output_neurons))
    #Sigmoid Function
    def sigmoid (self,x):
        return 1/(1 + exp(-x))
    #Derivative of Sigmoid Function
    def derivatives_sigmoid(self,x):
        return x * (1 - x)
    # The neural network thinks. 
    def learn(self, inputs):
        h=self.sigmoid(dot(inputs,self.wh)+self.bh)
        o=self.sigmoid(dot(h,self.wout)+self.bout)                
        return o 
    # Train the neural network and adjust the weights each time.
    def train(self,X,Y,training_iterations):
        for i in range(training_iterations):
            #Forward Propogation
            hinp1=dot(X,self.wh)
            hinp=hinp1 + self.bh
            hlayer_act = self.sigmoid(hinp)
            outinp1 = dot(hlayer_act,self.wout)
            outinp = outinp1+ self.bout
            output = self.sigmoid(outinp)
            #Backpropagation
            EO = Y - output
            outgrad = self.derivatives_sigmoid(output)
            d_output = EO * outgrad
            EH = d_output.dot(self.wout.T)
            hiddengrad = self.derivatives_sigmoid(hlayer_act)
            #how much hidden layer wts contributed to error
            d_hiddenlayer = EH * hiddengrad
            self.wout += hlayer_act.T.dot(d_output) *self.lr
            # dotproduct of nextlayererror and currentlayerop
            self.wh += X.T.dot(d_hiddenlayer) *self.lr
        return output
#Initialize 
neural_network = NeuralNet() 
# The training set
inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
print("Input vector X:\n",inputs)
outputs =array([[1, 0, 1]]).T
print("Target output vector O:\n",outputs)
# Train the neural network
output=neural_network.train(inputs,outputs, 8000)
print("Predicted Output: \n" ,output)
# Test the neural network with a test example. 
print("Output of BNN for the test sample is:\n",neural_network.learn(array([1,0,0])) )