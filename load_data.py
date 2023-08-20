import os 
import scipy.io as sio

def loadData(name):
    print("Loading Data ....................................")
    if name == 'IP':
        print("Indian_pines Data ....................................")
        data = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
    elif name == 'S':
        print("Salinas Data ....................................")
        data = sio.loadmat('Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('Salinas_gt.mat')['salinas_gt']
    elif name == 'P':
        print("Pavia Data ....................................")
        data = sio.loadmat('Pavia.mat')['pavia']
        labels = sio.loadmat('Pavia_gt.mat')['pavia_gt']
    elif name == 'SA':
        # print(sio.loadmat('SalinasA_corrected.mat'))
        print("SalinasA Data ....................................")
        data = sio.loadmat('SalinasA_corrected.mat')['salinasA_corrected']
        labels = sio.loadmat('SalinasA_gt.mat')['salinasA_gt']
    elif name == 'PU':
        print("PaviaU Data ....................................")
        data = sio.loadmat('PaviaU.mat')['paviaU']
        labels = sio.loadmat('PaviaU_gt.mat')['paviaU_gt']
        
    return data, labels, name