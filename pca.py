import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh, inv
from numpy import cov

def readData():
    df = pd.read_csv("D://Machine-Learning//testdata.csv",
                     index_col = 0)
    
    df.index.names = ['Country']
    df.columns.names = ['Year']
    return df

def preparePlot( xticks, 
                 yticks, 
                 figsize=(10.5, 6), 
                 hideLabels=False, 
                 gridColor='#999999',
                 gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax


def plot( example, df ):
    plt.close()
    fig, ax = preparePlot(np.arange(1990, 2008), np.arange(-.1, .6, .1))
    ax.set_xlabel(r'year'), ax.set_ylabel(r'values')

    x = map( lambda x : int( x ), df.columns.values )
    plt.plot( x, np.asarray( example ), c='#8cbfd0', linewidth='3.0')
    plt.show()
    
def parse(df):
    i = df.index.values
    x = map( lambda i : df.ix[i] , i )
    z = map( lambda y : y.tolist(), x )
    return zip(i,z)

def toFloat(x):
    try:
        a = float(x)
    except ValueError:
        return float( x.replace( ',' , '' ) )
    else:
        return a
    
def getmean( data ):
    return np.mean(data, axis=0)

def extract( ex ):
    ex = map( lambda x : toFloat( x ) , ex[1])
    mean = getmean( ex )
    exdemeaned =  map( lambda x : (x - mean)/mean, ex )
    return exdemeaned

def rescale(conformantdata, df):
    i = df.index.values
    scaledData = map( lambda v : extract(v), conformantdata)
    return zip(i,scaledData)

    """Compute the covariance matrix for a given dataset.

    """
def estimateCovariance( data ):
    print data
    mean = getmean( data )
    print mean
    dataZeroMean = map(lambda x : x - mean, data )
    print dataZeroMean
    covar = map( lambda x : np.outer(x,x) , dataZeroMean )
    print getmean( covar ) 
    return getmean( covar )

    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    """
def pca(data, k=2):
    
    d = estimateCovariance(  data )
    
    eigVals, eigVecs = eigh(d)

    validate( eigVals, eigVecs )
    inds = np.argsort(eigVals)[::-1]
    topComponent = eigVecs[:,inds[:k]]
    print '\nTop Component: \n{0}'.format(topComponent)
    
    correlatedDataScores = map(lambda x : np.dot( x ,topComponent), data )
    print ('\nScores : \n{0}'
       .format('\n'.join(map(str, correlatedDataScores))))
    print '\n eigenvalues: \n{0}'.format(eigVals[inds])
    return topComponent,correlatedDataScores,eigVals[inds]


    """Mutiply the eigen vector, its inverse and a diagonal matrix of eign values

    """
def  validate( eigVals, eigVecs ):
    print eigVecs    
    m = np.matrix( np.diag( eigVals ) )
    print '\n Validation: \n{0}'.format( np.dot( np.dot( eigVecs, inv( eigVecs ) ), m ) )
        
if __name__=="__main__":
    df = readData()
    conformantdata = parse( df )
    scaledData = rescale(conformantdata, df)
    example = filter(lambda (k, v): np.std(v) > 0.1, scaledData )
    b = map(lambda (k, v): v , scaledData )
    b = np.asarray(b)
    

    componentsScaled, scaledScores, eigenvaluesScaled = pca( b )
#     pcatestdata = [np.arange(x, x + 4) for x in np.arange(0, 20, 4)]   
#     componentsScaled, scaledScores, eigenvaluesScaled = pca( pcatestdata )
