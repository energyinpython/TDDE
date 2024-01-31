import os
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

from pyrepo_mcda.mcda_methods import TOPSIS
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import normalizations as norms
from daria import DARIA


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value


# plot line chart for sensitivity analysis
def plot_lineplot_sensitivity(data_sens, title = ""):
    """
    Visualization method to display line chart of alternatives rankings obtained with 
    modification of weight of given criterion.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different weight of
            selected criterion. The particular rankings are contained in subsequent columns of 
            DataFrame.

    Examples
    ----------
    >>> plot_lineplot_sensitivity(df_plot)
    """
    plt.figure(figsize = (9, 6))
    for j in range(data_sens.shape[0]):
        
        plt.plot(data_sens.iloc[j, :], linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(data_sens.index[j], (x_max, data_sens.iloc[j, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel("Weights", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.yticks(np.arange(1, 30, 2), fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle = ':')
    plt.tight_layout()
    
    plt.savefig('./results_sens_weights/' + 'sensitivity_' + title + '.png')
    plt.show()



def main():
    path = 'DATASET'
    # Number of countries
    m = 28

    # Symbols of Countries
    coun_names = pd.read_csv('DATASET/country_names.csv')
    country_names = list(coun_names['Symbol'])

    str_years = [str(y) for y in range(2018, 2022)]
    # loop
    # dataframe for annual results TOPSIS

    vect_weights = np.arange(0.1, 0.9, 0.1)

    # ===============================================================================
    # Macrolevel criteria
    pref_sens = pd.DataFrame(index = country_names)
    rank_sens = pd.DataFrame(index = country_names)

    for vw in vect_weights:

        weights = np.zeros(8)
        weights[[0, 1, 2, 3]] = vw / 4
        weights[[4, 5, 6, 7]] = (1 - vw) / 4

        preferences_t = pd.DataFrame(index = country_names)
        rankings_t = pd.DataFrame(index = country_names)

        # initialization of the TOPSIS method object
        topsis = TOPSIS(normalization_method=norms.max_normalization)

        for el, year in enumerate(str_years):

            file = 'eides_' + str(year) + '.csv'
            pathfile = os.path.join(path, file)
            data = pd.read_csv(pathfile, index_col = 'Country')
            
            # types: only profit 1 (performances)
            types = np.ones(data.shape[1])
            
            list_of_cols = list(data.columns)
            # decision matrix
            matrix = data.to_numpy()

            print(np.sum(weights))
            # TOPSIS annual
            pref_t = topsis(matrix, weights, types)
            rank_t = rank_preferences(pref_t, reverse = True)
            
            preferences_t[year] = pref_t
            rankings_t[year] = rank_t
        
        # ======================================================================
        # DARIA-TOPSIS method
        # ======================================================================
        # DARIA (DAta vaRIAbility) temporal approach
        # preferences includes preferences of alternatives for evaluated years
        df_varia_fin = pd.DataFrame(index = country_names)
        df = preferences_t.T
        matrix = df.to_numpy()

        # TOPSIS orders preferences in descending order
        met = 'topsis'
        type = 1

        # calculate efficiencies variability using DARIA methodology
        daria = DARIA()
        # calculate variability values with Entropy
        var = daria._entropy(matrix)
        # calculate variability directions
        dir_list, dir_class = daria._direction(matrix, type)

        # for next stage of research
        df_varia_fin[met.upper()] = list(var)
        df_varia_fin[met.upper() + ' dir'] = list(dir_class)

        df_results = pd.DataFrame()
        df_results['Ai'] = list(df.columns)
        df_results['Variability'] = list(var)
        
        # list of directions
        df_results['dir list'] = dir_list
        
        df_varia_fin = df_varia_fin.rename_axis('Ai')
        

        # final calculation
        # data with alternatives' rankings' variability values calculated with Gini coeff and directions
        G_df = copy.deepcopy(df_varia_fin)

        # data with alternatives' efficiency of performance calculated for the recent period
        S_df = copy.deepcopy(preferences_t)

        # ==============================================================
        # S = S_df.mean(axis = 1).to_numpy()
        S = S_df['2021'].to_numpy()

        G = G_df[met.upper()].to_numpy()
        dir = G_df[met.upper() + ' dir'].to_numpy()

        # update efficiencies using DARIA methodology
        # final updated preferences
        final_S = daria._update_efficiency(S, G, dir)

        # TOPSIS has descending ranking from prefs
        rank = rank_preferences(final_S, reverse = True)

        rank_sens[str(np.round(vw, 1))] = rank

    print(rank_sens)
    rank_sens.to_csv('results_sens_weights/' + 'macrolevel' + '.csv')
    plot_lineplot_sensitivity(rank_sens, title = "Modification of Macrolevel criteria weights")


    # ===============================================================================
    # Microlevel criteria
    pref_sens = pd.DataFrame(index = country_names)
    rank_sens = pd.DataFrame(index = country_names)

    for vw in vect_weights:

        weights = np.zeros(8)
        weights[[4, 5, 6, 7]] = vw / 4
        weights[[0, 1, 2, 3]] = (1 - vw) / 4
        
        preferences_t = pd.DataFrame(index = country_names)
        rankings_t = pd.DataFrame(index = country_names)

        # initialization of the TOPSIS method object
        topsis = TOPSIS(normalization_method=norms.max_normalization)

        for el, year in enumerate(str_years):

            file = 'eides_' + str(year) + '.csv'
            pathfile = os.path.join(path, file)
            data = pd.read_csv(pathfile, index_col = 'Country')
            
            # types: only profit 1 (performances)
            types = np.ones(data.shape[1])
            
            list_of_cols = list(data.columns)
            # decision matrix
            matrix = data.to_numpy()

            print(np.sum(weights))
            # TOPSIS annual
            pref_t = topsis(matrix, weights, types)
            rank_t = rank_preferences(pref_t, reverse = True)
            
            preferences_t[year] = pref_t
            rankings_t[year] = rank_t
        
        # ======================================================================
        # DARIA-TOPSIS method
        # ======================================================================
        # DARIA (DAta vaRIAbility) temporal approach
        # preferences includes preferences of alternatives for evaluated years
        df_varia_fin = pd.DataFrame(index = country_names)
        df = preferences_t.T
        matrix = df.to_numpy()

        # TOPSIS orders preferences in descending order
        met = 'topsis'
        type = 1

        # calculate efficiencies variability using DARIA methodology
        daria = DARIA()
        # calculate variability values with Entropy
        var = daria._entropy(matrix)
        # calculate variability directions
        dir_list, dir_class = daria._direction(matrix, type)

        # for next stage of research
        df_varia_fin[met.upper()] = list(var)
        df_varia_fin[met.upper() + ' dir'] = list(dir_class)

        df_results = pd.DataFrame()
        df_results['Ai'] = list(df.columns)
        df_results['Variability'] = list(var)
        
        # list of directions
        df_results['dir list'] = dir_list
        
        df_varia_fin = df_varia_fin.rename_axis('Ai')
        

        # final calculation
        # data with alternatives' rankings' variability values calculated with Gini coeff and directions
        G_df = copy.deepcopy(df_varia_fin)

        # data with alternatives' efficiency of performance calculated for the recent period
        S_df = copy.deepcopy(preferences_t)

        # ==============================================================
        # S = S_df.mean(axis = 1).to_numpy()
        S = S_df['2021'].to_numpy()

        G = G_df[met.upper()].to_numpy()
        dir = G_df[met.upper() + ' dir'].to_numpy()

        # update efficiencies using DARIA methodology
        # final updated preferences
        final_S = daria._update_efficiency(S, G, dir)

        # TOPSIS has descending ranking from prefs
        rank = rank_preferences(final_S, reverse = True)

        rank_sens[str(np.round(vw, 1))] = rank

    print(rank_sens)
    rank_sens.to_csv('results_sens_weights/' + 'microlevel' + '.csv')
    plot_lineplot_sensitivity(rank_sens, title = "Modification of Microlevel criteria weights")


    # ===============================================================================
    # Particular criteria

    for ind in range(0, 8):

        pref_sens = pd.DataFrame(index = country_names)
        rank_sens = pd.DataFrame(index = country_names)

        for vw in vect_weights:

            rest = 1 - vw

            weights = np.ones(8) * ((1 - vw) / 7)
            weights[ind] = vw

            preferences_t = pd.DataFrame(index = country_names)
            rankings_t = pd.DataFrame(index = country_names)

            # initialization of the TOPSIS method object
            topsis = TOPSIS(normalization_method=norms.max_normalization)

            for el, year in enumerate(str_years):

                file = 'eides_' + str(year) + '.csv'
                pathfile = os.path.join(path, file)
                data = pd.read_csv(pathfile, index_col = 'Country')
                
                # types: only profit 1 (performances)
                types = np.ones(data.shape[1])
                
                list_of_cols = list(data.columns)
                # decision matrix
                matrix = data.to_numpy()

                print(np.sum(weights))
                # TOPSIS annual
                pref_t = topsis(matrix, weights, types)
                rank_t = rank_preferences(pref_t, reverse = True)
                
                preferences_t[year] = pref_t
                rankings_t[year] = rank_t
            
            # ======================================================================
            # DARIA-TOPSIS method
            # ======================================================================
            # DARIA (DAta vaRIAbility) temporal approach
            # preferences includes preferences of alternatives for evaluated years
            df_varia_fin = pd.DataFrame(index = country_names)
            df = preferences_t.T
            matrix = df.to_numpy()

            # TOPSIS orders preferences in descending order
            met = 'topsis'
            type = 1

            # calculate efficiencies variability using DARIA methodology
            daria = DARIA()
            # calculate variability values with Entropy
            var = daria._entropy(matrix)
            # calculate variability directions
            dir_list, dir_class = daria._direction(matrix, type)

            # for next stage of research
            df_varia_fin[met.upper()] = list(var)
            df_varia_fin[met.upper() + ' dir'] = list(dir_class)

            df_results = pd.DataFrame()
            df_results['Ai'] = list(df.columns)
            df_results['Variability'] = list(var)
            
            # list of directions
            df_results['dir list'] = dir_list
            
            df_varia_fin = df_varia_fin.rename_axis('Ai')
            

            # final calculation
            # data with alternatives' rankings' variability values calculated with Gini coeff and directions
            G_df = copy.deepcopy(df_varia_fin)

            # data with alternatives' efficiency of performance calculated for the recent period
            S_df = copy.deepcopy(preferences_t)

            # ==============================================================
            # S = S_df.mean(axis = 1).to_numpy()
            S = S_df['2021'].to_numpy()

            G = G_df[met.upper()].to_numpy()
            dir = G_df[met.upper() + ' dir'].to_numpy()

            # update efficiencies using DARIA methodology
            # final updated preferences
            final_S = daria._update_efficiency(S, G, dir)

            # TOPSIS has descending ranking from prefs
            rank = rank_preferences(final_S, reverse = True)

            rank_sens[str(np.round(vw, 1))] = rank

        print(rank_sens)
        rank_sens.to_csv('results_sens_weights/C' + str(ind + 1) + '.csv')
        plot_lineplot_sensitivity(rank_sens, title = "Modification of criterion " + r'$C_{' + str(ind + 1) + '}$')
        
    
if __name__ == '__main__':
    main()