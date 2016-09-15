###
# This code provides a way to approximate the probability of
# finding two features together using von Neumann Diffusion Kernel.
# by: Umut Eser, 09/15/2016
#
# required modules:
# scipy, numpy, matplotlib, pandas, seaborn
###

from scipy.linalg import inv, eigvals
import numpy as np
from matplotlib import pylab as pl
import pandas as pd # for data frame handling
import seaborn as sns # for pretty plotting
from optparse import OptionParser # to run in the terminal


usage = 'usage: %prog [options] <path_to_data>'
parser = OptionParser(usage)
(options,args) = parser.parse_args()

def main():
    df = pd.read_csv(args[0])
    D = vonNeumannDiffKernel(df)
    cg = sns.clustermap(D,cmap='afmhot');
    pl.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);
    pl.savefig('vonNeumannDiffusionKernelPairProbability.pdf')

def vonNeumannDiffKernel(B):
    '''
    Calculates the pair probability using von Neumann diffusion kernel

    inputs
    B : a 2D numpy array (or pandas dataframe) where the columns are the features, rows are the samples

    returns
    D : a symmetric 2D numpy array (or pandas dataframe) whose entries, D_ij correspond to the probability of
    finding ith and jth feature together

    '''
    features = B.columns
    A = np.matmul(B.T,B)
    rho = max(abs(eigvals(A)))
    kappa = np.sum(np.sum(A>0)/(A.shape[0]*A.shape[1]))
    gamma = 1/((1+kappa)*rho)
    K = np.matmul(A,inv(np.identity(A.shape[0])-gamma*A))
    D = np.zeros_like(K)

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            D[i,j] = K[i,j]/np.sqrt(K[i,i]*K[j,j]+1e-16)

    return pd.DataFrame(D,columns=features, index=features)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
