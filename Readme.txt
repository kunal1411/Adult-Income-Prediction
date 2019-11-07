To run the file ML _Income_Assignment_2.ipynb
 
## Download adult.csv as attached in the zip file submitted

To run the file ML HW2 Energy.ipynb

## Download energydata_complete.csv from https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

## import the following libraries
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.plotly as py
import plotly.graph_objs as go
import warnings
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns
import os
import time

#Setting options
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")

#Set Default option
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
pd.options.display.max_rows = 3000

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import sklearn as sk
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score