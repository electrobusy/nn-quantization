
import numpy as np
import matplotlib.pyplot as plt
import math

"""
1 - Generate sine way
"""
# We'll generate this many sample datapoints
SAMPLES = 1000

# Set a "seed" value, so we get the same random numbers each time we run this
# notebook. Any number can be used here.
SEED = 1337
np.random.seed(SEED)

# Generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)

# Shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)

# Calculate the corresponding sine values
y_values = np.sin(x_values)

# Plot our data. The 'b.' argument tells the library to print blue dots.
plt.plot(x_values, y_values, 'b.')
plt.xlabel('x [-]')
plt.ylabel('sin(x) [-]')
plt.grid()
# plt.show()

"""
2 - Add noise in the data
"""
# Add a small random number to each y value
y_values += 0.1 * np.random.randn(*y_values.shape)

# Plot our data
plt.plot(x_values, y_values, 'b.')
plt.xlabel('x [-]')
plt.ylabel('sin(x) [-]')
plt.grid()
# plt.show()

"""
3 - split data in training/validation/testing sets
"""
# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
TRAIN_SPLIT =  int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_val, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_val, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_val.size + x_test.size) ==  SAMPLES

# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_val, y_val, 'y.', label="Val")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.legend()
plt.xlabel('x [-]')
plt.ylabel('sin(x) [-]')
plt.grid()
# plt.show()

"""
4 - Create and train network
"""
import time 

import torch 
from torch import nn, optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --
# Function created to make model
# --
def mlp(layer_dims):

    # model operations
    model_ops = list()

    # number of layers
    nl = len(layer_dims)

    # add operations
    for i in range(nl - 1):
        
        # model_ops.append(torch.nn.LayerNorm(layer_dims[i]))
        
        # linear layer
        a = torch.nn.Linear(layer_dims[i], layer_dims[i + 1])
        # -- apply xavier initilization to the weights
        torch.nn.init.xavier_uniform_(a.weight)
        # -- biases should be zero
        torch.nn.init.zeros_(a.bias)
        model_ops.append(a)

        # if penultimate layer
        # if i < nl - 2:
        #     # output between -1 and 1
        #     model_ops.append(torch.nn.Tanh())
        #     pass

        # if hidden layer
        if i < nl - 2:
            # activation hidden layer
            model_ops.append(torch.nn.Softplus())
            pass

    # create sequential model
    model = torch.nn.Sequential(*model_ops)

    return model

# training parameters
# LR = 1e-4
MAX_EPOCH = 600
BATCH_SIZE = 16

# network architecture parameters
HL = 1
HN = 16

# model 
layer_dims = [1] + [HN]*HL + [1]
model = mlp(layer_dims)

# print(model)

# optimizer
opt = optim.RMSprop(model.parameters())

# loss function
lf = nn.MSELoss()

# convert dataset from numpy array to torch array
x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)

x_val = torch.from_numpy(x_val).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)

x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)


# divide dataset in batches using DataLoader
train_dataset = torch.utils.data.TensorDataset(x_train.unsqueeze(1),y_train.unsqueeze(1))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(x_val.unsqueeze(1),y_val.unsqueeze(1))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(x_test.unsqueeze(1),y_test.unsqueeze(1))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
 
# list to store different losses
ltrn_list, lval_list, ltst_list = list(), list(), list()

""" 

print('============================')
print('Train network...')
start_time = time.time()
# iterate through episodes
for e in range(MAX_EPOCH):

    # episode loss
    msetrn = 0
    mseval = 0
    msetst = 0
 
    for itrn_batch, otrn_batch in train_loader:
            
        # zero gradients
        opt.zero_grad()

        # training loss
        ltrn_batch = lf(model(itrn_batch), otrn_batch)

        # backpropagate training error
        ltrn_batch.backward()

        # update weights
        opt.step()

        # track losses
        msetrn += ltrn_batch.item()

    with torch.no_grad():
        for ival_batch, oval_batch in val_loader:

            # validation loss
            lval_batch = lf(model(ival_batch), oval_batch)

            # track losses
            mseval += lval_batch.item()
        
    # print progress
    ltrn_list.append(msetrn/len(train_loader))
    lval_list.append(mseval/len(val_loader))
    print("Epoch {} / Train Loss: {} / Val Loss: {}".format(e+1, ltrn_list[-1], lval_list[-1]))
    
    with torch.no_grad():
        for itst_batch, otst_batch in test_loader:

            # testing loss
            ltst_batch = lf(model(itst_batch), otst_batch)

            # track losses
            msetst += ltst_batch.item() 

    ltst_list.append(msetst/len(test_loader))

end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
    
losses = {
    'train': ltrn_list,
    'val': lval_list,
    'test': ltst_list,
}

epochs = range(1, MAX_EPOCH + 1)
    
plt.clf()
plt.plot(epochs, losses['train'], 'g', label='Training loss')
plt.plot(epochs, losses['val'], 'b', label='Validation loss')
plt.plot(epochs, losses['test'], 'r', label='Testing loss')
plt.title('Training/Validation/Testing loss')
plt.xlabel('Epoch [-]')
plt.ylabel('Loss ' + '(' + 'MSE' + ')' + ' [-]')
plt.grid()
plt.legend()
plt.show()

""" 

""" 
import os 
plotName = 'loss_plot_' + str(MAX_EPOCH) + '_epochs' + '.png'
plt.savefig(plotName)
print("Plot '%s' created successfully" %plotName)
""" 

# torch.save(model.state_dict(),"sine_model_" + str(HL) + "_HL_" + str(HN) + "_HN_" + str(600) + "_epochs.pt")

# Load model
model = mlp(layer_dims).to('cpu')
model.load_state_dict(torch.load("sine_model_" + str(HL) + "_HL_" + str(HN) + "_HN_" + str(600) + "_epochs.pt"))

# print(model)

"""
5 - check prediction (visually)
""" 
model.eval()
data = torch.from_numpy(x_values).float().to(device).unsqueeze(1) # Load your data here, this is just dummy data
output = model(data)

plt.clf()
plt.plot(x_values, y_values, 'g.', label='GT')
plt.plot(x_values, output.detach().numpy(), 'b.', label='NN')
plt.xlabel('x [-]')
plt.ylabel('sin(x) [-]')
plt.grid()
plt.legend()
# plt.show()




