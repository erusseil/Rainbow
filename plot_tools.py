import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


band_dict = {0:'g ', 1:'r ', 2:'i '}

markers_train = ["o", "X"]
markers_test = ["D", "^"]

def draw_confusion(clf, X_test, y_test, interest, percent=False):
    
    classe_names = np.unique(y_test)

    fig = plt.figure(figsize=(13,11))
    ax = fig.add_subplot(111)

    # On calcul la matrice de confusion de nos fonctions
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test), normalize=percent)

    #Show the conf matrix of Bazin
    c = sns.heatmap(conf_matrix, xticklabels=classe_names, yticklabels=classe_names, annot=True,cbar=True, cmap="mako",linewidths=1,annot_kws={'size': 20});
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 14)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 14)

    plt.xlabel("\nPredicted label",fontsize = 16)
    plt.ylabel("Real label",fontsize = 16)

    pos_interest = np.argwhere(np.array(classe_names) == interest)[0,0]
    eff = conf_matrix[pos_interest,pos_interest]/conf_matrix[pos_interest,:].sum()
    purity = conf_matrix[pos_interest,pos_interest]/conf_matrix[:,pos_interest].sum()
    print(f'Efficiency : {eff*100:.2f}% of real {interest} were labeled as {interest}')
    print(f'Purity : {purity*100:.2f}% of classified {interest} were indeed {interest}')

    plt.show()
    
def zoom_in(df, zoom, parameters):
 
    """ Create a mask that excludes extreme values. Allow for clarity in scatter matrix plot
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing all parameters to plot
        
    zoom: float
        Proportion of non extreme value points to keep
        
    parameters : list
        List of all parameter names for which a zoom-in is wanted
        
    
    
    Returns
    -------
    
    np.array
        Mask to apply to df to filter extreme values
        
    """
    to_exclude = np.array([])
    for p in parameters:

        arr = df[f'{p}'].copy()
        # Exclude most extreme values (both minimal and maximal) : cut is the nb of obj to exclude in extremal values
        cut = round((len(arr) - zoom*len(arr))/2)

        # Sort the values
        arr = arr.sort_values()
        # Get a list of all the extreme points
        arr = arr.iloc[:cut].append(arr.iloc[len(arr)-cut:])

        # Add the indexes of the points to a list
        to_exclude = np.append(to_exclude, arr.index)

    mask = (df.index).isin(to_exclude)

    return ~mask


class Standard_plot():
    
    yellow_snad = '#fcbd43'
    dark_snad = '#22114c'
    third_color = '#C45C3F'

    def __init__(self, n=1, m=1):
        self.title_size = 20
        self.figure_size = (10,6)
        self.xylabel_size = 18
        self.legend_size = 16
        self.marker_size = 100
        self.err_marker_size = 13
        self.line_size = 6
        self.err_line_size = 3
        self.axis_size = 2
        self.tick_size = 14
        self.loc = 'upper right'
        self.n, self.m = n, m
        self.fig, self.axes = None, None
    
    def create_fig(self):
        # change all spines
        
        self.fig, self.axes = plt.subplots(self.n, self.m, figsize=self.figure_size)
        
        if (self.n==1) & (self.m==1):
            self.axes = [self.axes]
        if (self.n!=1) & (self.m!=1):
            self.axes = self.axes.flatten()
        
        for axis in ['top','bottom','left','right']:
            for ax in self.axes:
                ax.spines[axis].set_linewidth(self.axis_size)

        # increase tick width
        for ax in self.axes:
            ax.tick_params(width=self.axis_size)
        
    def end_fig(self, idx=-1, legend=True):
        
        if idx==-1:
            plt.yticks(fontsize=self.tick_size)
            plt.xticks(fontsize=self.tick_size)
            plt.legend(fontsize = self.legend_size, loc=self.loc)
        else:
            self.axes[idx].tick_params(labelsize=self.tick_size)
            if legend:
                self.axes[idx].legend(fontsize = self.legend_size, loc=self.loc)