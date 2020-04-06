import torch
import matplotlib.pyplot as plt 

from sine_mlp_train import mlp

if __name__ == '__main__':
    # network architecture parameters
    HL = 1
    HN = 16
    layer_dims = [1] + [HN]*HL + [1]

    """
    6 - Load model
    """
    model = mlp(layer_dims).to('cpu')
    model.load_state_dict(torch.load("sine_model_" + str(HL) + "_HL_" + str(HN) + "_HN_" + str(600) + "_epochs.pt")) 
    model.eval()

    print(model[0].bias.detach().numpy())

    """
    7 - Plot histogram with network weights
    """
    nbins = 25
    fig, ax = plt.subplots(2,2)
    ax[0,0].hist(model[0].weight.detach().numpy(),nbins)
    ax[0,0].grid()
    ax[0,0].set_xlabel('weights [-]')
    ax[0,0].set_ylabel('# of occurences [-]')

    ax[0,1].hist(model[0].bias.detach().numpy(),nbins)
    ax[0,1].grid()
    ax[0,1].set_xlabel('bias [-]')
    ax[0,1].set_ylabel('# of occurences [-]')

    ax[1,0].hist(model[0].weight.detach().numpy(),nbins)
    ax[1,0].grid()
    ax[1,0].set_xlabel('weights [-]')
    ax[1,0].set_ylabel('# of occurences [-]')

    ax[1,1].hist(model[2].bias.detach().numpy(),nbins)
    ax[1,1].grid()
    ax[1,1].set_xlabel('bias [-]')
    ax[1,1].set_ylabel('# of occurences [-]')

    fig.tight_layout()
    plt.show()

    """
    
    """
