import os
import numpy as np
import pickle
from datetime import timedelta
import pandas as pd

def preprocess_RFM(df):
    """
    To agregate customer orders implementing RFM method.
    Recence start date is the day after the latest invoice date.
    """

    temp = df.copy()
    temp['TotalPrice'] = temp['Quantity'] * temp['UnitPrice']
    d_study = temp.InvoiceDate.max() + timedelta(days=1)
    temp = temp.groupby('CustomerID').\
                agg(# récence
                    {'InvoiceDate': [('recency', \
                                      lambda x: (d_study - x.max()).days),
                                     # fréquence
                                     ('frequency', 'count')],
                     # montant total des commandes
                     'TotalPrice': [('monetary_value', 'sum')]
                     }).round(2).reset_index()
    temp.columns = ['CustomerID', 'recency', 'frequency', 'monetary_value']

    return temp

def segment_customers(df):
    """
    Enable to 'labelize' the customers as Platinum/Gold/Silver/Bronze 
    after calculating their R/F/M values from dataframe df 
    """

    with open('OC_DS_P5.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        clf = unpickler.load()

    temp = preprocess_RFM(df)
    X = temp[['recency', 'frequency', 'monetary_value']]
    X_scaled = preprocessing.StandardScaler().fit_transform(np.log(X))
    temp = temp.assign(Segment=clf.predict(X_scaled) )

    return temp
