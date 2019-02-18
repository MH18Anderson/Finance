import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas.io.data as web
from sklearn.decomposition import KernelPCA

def pca(tickers):
    '''
    Calculates which stocks are most correlated to an index (which stocks can best explain
    the index movements), by deriving principal components of the index

    Will plot most important component and top 5 components

    Tickers should be inputted as a list, using the symbols from yahoo finance
    The index of interest should be entered as the last element in the list
    i.e. tickers = ['SAP.DE', 'BAYN.DE', 'BMW.DE', '^GDAXI']
    notice '^GDAXI' is the symbol for the DAX

    '''
    index = tickers[-1]

    data = pd.DataFrame()
    for tick in tickers:
        data[tick] = web.DataReader(tick, data_source = 'yahoo')['Close']
    data = data.dropna() # clean the data

    # separate out the index data
    base_index = pd.DataFrame(data.pop(index))

    # pca usually works with normalized data sets
    scale_function = lambda x: (x - x.mean()) / x.std()
    pca = KernelPCA().fit(data.apply(scale_function))

    # pca analysis produces many components - most of which have negligible importance
    # so we restrict our analysis to the most important components
    # since we are interested in the relative importance of each component, normalize again
    main_ten = lambda x: x / x.sum()

    # pca index with a single component only first
    pca = KernelPCA(n_components=1).fit(data.apply(scale_function))
    base_index['PCA_1'] = pca.transform(-data)

    base_index.apply(scale_function).plot(figsize=(8, 4))

    # pca with 5 components
    # calculate a weighted average from the single resulting components
    pca = KernelPCA(n_components=5).fit(data.apply(scale_function))
