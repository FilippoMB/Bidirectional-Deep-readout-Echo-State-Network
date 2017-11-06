# General imports
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from modules import train_ESN, train_RNN, train_BDESN
from reservoir import Reservoir

dataset_name = 'LIB' # name of the dataset to process
n_runs = 2  # number of different random initializations for each method
use_seed = False  # set to False to generate different random initializations at each execution
plot_on = True  # set to false for textual output only

# Set True to train a classifier based on a given network
TRAIN_ESN = True
TRAIN_RNN = True
TRAIN_BDESN = True

# ------ Hyperparameters ------
# Parameters for ESN and BDESN
n_internal_units = 1000  # size of the reservoir
connectivity = 0.33  # percentage of nonzero connections in the reservoir
spectral_radius = 1.12  # largest eigenvalue of the reservoir
input_scaling = 0.47  # scaling of the input weights
noise_level = 0.07  # noise in the reservoir state update

# Parameters for GRU and BDESN
batch_size = 25  # samples in the mini-batches in gradient descent training
num_epochs = 5000  # number of epochs 
n_hidden_1 = 20  # size of 1st layer in MLP
n_hidden_2 = 20  # size of 2nd layer in MLP
n_hidden_3 = 10  # size of 3rd layer in MLP

# Parameters specific to ESN
w_ridge = 2.12  # Regularization coefficient for ridge regression

# Parameters specific to GRU
cell_type = 'GRU'  # type of cell in the recurrent layer. Available options are 'RNN', 'GRU' and 'LSTM'
num_cells = 30  # size of the recurrent layer
learning_rate_rnn = 0.001  # learning rate in Adam optimizer
w_l2_rnn = 0.00022  # L2 regularization weight in loss function
p_drop_rnn = 0.8  # dropout (keep) probability in MLP

# Parameters specific to BDESN
learning_rate_bdesn = 0.001  # learning rate in Adam optimizer
w_l2_bdesn = 0.0022  # L2 regularization weight in loss function
p_drop_bdesn = 0.95  # dropout (keep) probability in MLP
embedding_method = 'pca'  # dimensionality reduction method. Available options are 'identity', 'pca' and 'kpca'
n_dim = 30  # size of the space of reduced dimensionality

# Load dataset
data = scipy.io.loadmat('../dataset/'+dataset_name+'.mat')
X = data['X']  # shape is [N,T,V]
if len(X.shape) < 3:
    X = np.atleast_3d(X)
Y = data['Y']  # shape is [N,1]
Xte = data['Xte']
if len(Xte.shape) < 3:
    Xte = np.atleast_3d(Xte)
Yte = data['Yte']

# one-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse=False)
Y = onehot_encoder.fit_transform(Y)
Yte = onehot_encoder.transform(Yte)
num_classes = Y.shape[1]

# Output structures (ESN:0, RNN:1, BDESN:2)
training_time = np.zeros((n_runs, 3))
accuracy_list = np.zeros((n_runs, 3))
f1_list = np.zeros((n_runs, 3))
max_batches = X.shape[0]//batch_size

# Track the training loss (RNN:0, BDESN:1)
loss_track = np.zeros((2, int(np.ceil(max_batches*num_epochs))))

for r in range(n_runs):
    print('Executing run ', r+1, ' of ', n_runs, '...')

    # Set seed for PRNG
    seed = r if use_seed else None
    np.random.seed(seed)

    # Initialize common reservoir
    reservoir = Reservoir(n_internal_units, spectral_radius, connectivity,
                          input_scaling, noise_level)

    if TRAIN_ESN:
        print('--- Training ESN ---')
        accuracy_list[r, 0], f1_list[r, 0], training_time[r, 0] =  \
            train_ESN(X=X,
                      Y=Y,
                      Xte=Xte,
                      Yte=Yte,
                      n_internal_units=n_internal_units,
                      spectral_radius=spectral_radius,
                      connectivity=connectivity,
                      input_scaling=input_scaling,
                      noise_level=noise_level,
                      embedding_method='identity',
                      n_dim=None,
                      w_ridge=w_ridge,
                      reservoir=reservoir)
        
        print('\tTot training time: %.3f' % (training_time[r, 0]))
        print('\tAcc: %.3f, F1: %.3f' % (accuracy_list[r, 0], f1_list[r, 0]))

    if TRAIN_RNN:
        print('--- Training RNN ---')
        loss_track[0, :], accuracy_list[r, 1], f1_list[r, 1], training_time[r, 1] = \
            train_RNN(X=X,
                      Y=Y,
                      Xte=Xte,
                      Yte=Yte,
                      num_cells=num_cells,
                      fc_layout=[n_hidden_1,
                                 n_hidden_2,
                                 n_hidden_3],
                      batch_size=batch_size,
                      num_epochs=num_epochs,
                      p_drop=p_drop_rnn,
                      w_l2=w_l2_rnn,
                      learning_rate=learning_rate_rnn,
                      cell_type='GRU',
                      seed=seed)
            
        print('\tTot training time: %.3f' % (training_time[r, 1]))
        print('\tAcc: %.3f, F1: %.3f' % (accuracy_list[r, 1], f1_list[r, 1]))

    if TRAIN_BDESN:
        print('--- Training BDESN ---')
        loss_track[1, :], accuracy_list[r, 2], f1_list[r, 2], training_time[r, 2] = \
            train_BDESN(X=X,
                        Y=Y,
                        Xte=Xte,
                        Yte=Yte,
                        n_internal_units=n_internal_units,
                        spectral_radius=spectral_radius,
                        connectivity=connectivity,
                        input_scaling=input_scaling,
                        noise_level=noise_level,
                        embedding_method=embedding_method,
                        n_dim=n_dim,
                        fc_layout=[n_hidden_1,
                                   n_hidden_2,
                                   n_hidden_3],
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        p_drop=p_drop_bdesn,
                        w_l2=w_l2_bdesn,
                        learning_rate=learning_rate_bdesn,
                        seed=seed,
                        reservoir=reservoir)
            
        print('\tTot training time: %.3f' % (training_time[r, 2]))
        print('\tAcc: %.3f, F1: %.3f' % (accuracy_list[r, 2], f1_list[r, 2]))


if plot_on:
    import brewer2mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Show results in a table
    pd.set_option('display.width', 200)
    df = pd.DataFrame(
        np.vstack((accuracy_list.mean(axis=0)*100,
                   accuracy_list.std(axis=0)*100,
                   f1_list.mean(axis=0),
                   f1_list.std(axis=0),
                   training_time.mean(axis=0))).T,
        ('ESN', 'GRU', 'BDESN'),
        ['Accuracy [%]', 'Accuracy (std.)', 'F1', 'F1 (std.)', 'Tr. time [m]'])
    
    df = df.sort_values('Accuracy [%]', ascending=False)
    print(df)
    
    if TRAIN_BDESN or TRAIN_RNN:
    
        # Get a colors matrix
        bmap = brewer2mpl.get_map('Set3', 'qualitative', 3)
        colors = bmap.mpl_colors
    
        # Evolution of the objective function
        fig=plt.clf()
        fig=plt.figure(figsize=(5.5,3))
        plt.xlabel('Iteration')
        plt.ylabel('Objective function')
        plt.yscale('log')
        plt.title('Training Loss')
        #plt.rcParams["figure.figsize"] = [5,3]
    
        if TRAIN_RNN:
            ith_loss = loss_track[0, :] / n_runs
            plt.plot(np.arange(loss_track.shape[1]),
                     ith_loss,
                     label='GRU',
                     color=colors[0],
                     alpha=0.5)
    
            plt.plot(np.arange(loss_track.shape[1]),
                     pd.DataFrame(ith_loss.reshape(-1)).ewm(span=15).mean(),
                     color=colors[0],
                     alpha=1.0)
    
        if TRAIN_BDESN:
            ith_loss = loss_track[1, :] / n_runs
            plt.plot(np.arange(loss_track.shape[1]),
                     ith_loss,
                     label='BDESN',
                     color=colors[2],
                     alpha=0.5)
    
            plt.plot(np.arange(loss_track.shape[1]),
                     pd.DataFrame(ith_loss.reshape(-1)).ewm(span=15).mean(),
                     color=colors[2],
                     alpha=1.0)
    
        
        plt.legend(loc='best', fontsize=10)
        plt.grid()        
        plt.savefig('Training_loss',format='pdf')
        plt.show()
        
else:
    
    print('ESN: acc=%.3f+/-%.3f, f1=%.3f+/-%.3f, time=%.3f' 
      %(np.mean(accuracy_list[:,0]), np.std(accuracy_list[:,0]), 
        np.mean(f1_list[:,0]), np.std(f1_list[:,0]), 
        np.mean(training_time[:,0])))
      
    print('GRU: acc=%.3f+/-%.3f, f1=%.3f+/-%.3f, time=%.3f' 
          %(np.mean(accuracy_list[:,1]), np.std(accuracy_list[:,1]), 
            np.mean(f1_list[:,1]), np.std(f1_list[:,1]), 
            np.mean(training_time[:,1])))
      
    print('BDESN: acc=%.3f+/-%.3f, f1=%.3f+/-%.3f, time=%.3f' 
          %(np.mean(accuracy_list[:,2]), np.std(accuracy_list[:,2]), 
            np.mean(f1_list[:,2]), np.std(f1_list[:,2]), 
            np.mean(training_time[:,2])))
            
