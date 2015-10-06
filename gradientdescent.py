import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def readData():
    df = pd.read_csv("D://Machine-Learning//ex1data1.txt" , header=None)
    df.columns = [ 'X', 'y']
    return df

def preparePlot(  
                 figsize=(10.5, 6), 
                 hideLabels=False, 
                 gridColor='#999999',
                 gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    return fig, ax

def prepareData(df):
    r,c = df.shape
    df['ones'] = np.ones( ( r, 1 ) )
    df[['ones']] = df[['ones']].astype(float)
    X  = df.reindex(columns=['ones', 'X'])
    theta = np.zeros( shape = ( 2, 1 ) );
    return X,theta

def plot( x, y, x1, y1 ):
    fig, ax = preparePlot()
    ax.set_xlabel(r'X'), ax.set_ylabel(r'y')
    plt.scatter( x, y, c='#8cbfd0', marker='o')
    plt.plot( x1, y1, c='green' )
    plt.show()
    
def computecost(X, y, theta):
    
    J = 0;
    r,c = X.shape
    
    J = ( np.dot( X , theta ) - np.asarray(y) ) / (2 * r )

    sumsquare =  np.sum( J ** 2)

    """Not the actual formula but the code above seems to miss something
       that requires multiplication by 2 """
    return sumsquare * 2

def gradientDescent( X,
                     y,
                     theta,
                     alpha = 0.01,
                     num_iters = 1500):

    r,c = X.shape
    
    for iter in range( 1, num_iters ):
        theta = theta - ( ( alpha * np.dot( X.T, ( np.dot( X , theta ).T - np.asarray(y) ).T ) ) / r )
    return theta

if __name__=="__main__":
    df = readData()
    X,theta = prepareData(df)
    
    computecost( X,  df[ 'y' ], theta )
    
    theta = gradientDescent( X,  df[ 'y' ], theta )
    
    plot( df[ 'X'], df[ 'y' ],X['X'], np.dot( X, theta ) )
