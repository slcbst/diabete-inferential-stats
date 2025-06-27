import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, ks_2samp, kstest
from scipy.stats import zscore
from scipy.stats import norm

from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.metrics import r2_score

# ** EDA ** ----------------------------------------------------------------------------------------------------------------
# ** Probability ** --------------------------------------------------------------------------------------------------------
# ** Statistical Tests ** --------------------------------------------------------------------------------------------------
# ** Multivariate Analysis ** ----------------------------------------------------------------------------------------------
# ** Statistical Modelisation ** -------------------------------------------------------------------------------------------
# ** Machine Learning Models ** --------------------------------------------------------------------------------------------
def display_distribution(df: pd.DataFrame, var: str, var_name):
    # Variables
    var_mean = df[var].mean()
    var_median = df[var].median()
    
    # Create the graph
    fig, ax = plt.subplots(1,2,figsize=(16,4))
    
    # 1. Plot the distribution
    ax[0].hist(df[var])
    ax[0].axvline(var_mean, color='green', label='mean')
    ax[0].axvline(var_median, color='orange', linestyle = '-', label='median')
    ax[0].legend()
    ax[0].set_title(f"Histogram of the {var_name} {var}")
    
    # 2. Plot the boxplot 
    ax[1].boxplot(df[var], orientation='horizontal', meanline=True, showmeans=True)
    ax[1].set_title(f"Boxplot of {var_name} {var}")
    fig.suptitle(f"Distribution of {var_name} {var}")
    plt.show()
    
    
# ** Probability ** --------------------------------------------------------------------------------------------------------
def display_histogramme_loi_normal(df, var, bins=20):
    data = df[var]
    data = data.dropna()
    # 1. Calculate the mean and standatd deviation 
    mu = data.mean()
    sigma = np.std(data)

    # 2. plot histogramme empirique
    plt.hist(data, bins=20, density=True, alpha=0.6, edgecolor='b')
    
    # 3. plot normal distribution 
    x = np.linspace(min(data), max(data), 1000)
    y = norm.pdf(x, loc=mu, scale=sigma)
    plt.plot(x,y, color='r', label='Loi Normal Théorique')
    
    plt.title(f"Histogramme de '{var}' et courbe de la loi normale")
    plt.xlabel(var)
    plt.ylabel('Densité')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_stats_normality(df, var , mu, sigma, alpha =0.05):
    # Variables, constantes
    data = df[var] 

    # Test normality
    ks_stat, ks_pvalue = kstest(data, 'norm', (mu, sigma))
    shapiro_stat, shapiro_pvalue = shapiro(data)

    if ks_pvalue > alpha:
        ks_interpretation = "fail to reject H0"
        ks_conclusion = f"- As the p_value > {alpha} for Kolmogorov-Smirnov test, so we do not reject H0 and assume that {var} distribution IS normal"
    else:
        ks_interpretation = "reject H0"
        ks_conclusion = f"- As the p_value < {alpha} for Kolmogorov-Smirnov test, so reject H0 and assume that {var} distribution IS NOT normal"

    if shapiro_pvalue > alpha:
        shapiro_interpretation = "fail to reject H0"
        shapiro_conclusion = "p_value > 0.05, we do not reject H0"
    else:
        shapiro_interpretation = "reject H0"
        shapiro_conclusion = "p_value < 0.05, we reject H0"

    dict_result = {'Test': ['Kolmogorov-Smirnov','Shapiro'],
                   'statistic': [ks_stat, shapiro_stat],
                   'p_value': [ks_pvalue, shapiro_pvalue],
                  'interpretation': [ks_interpretation, shapiro_interpretation]}
    
    print("\n **Tests d'adéquation à la loi normale - Normality Tests**")
    print(f"alpha = {alpha}")
    
    result = pd.DataFrame.from_dict(dict_result)
    conclusion = f"\nConclusion:\n {ks_conclusion}"
    return (result, conclusion)


def compare_normality_pre_post_outlier_removal(df, var , mu, sigma, alpha): 
    data = df[var] 
    # 1. Test the normality of the all observation
    result, conclusion = test_stats_normality(df, var,  mu, sigma, alpha)
    print(result)
    print(conclusion)

    # 2. Remove outlier with the z_score value
    print("\n**Removing Outliers**")
    print("\n Here, as we know the Shapiro test is sensitive to outlier, we will remove them (using the zcore method)")
    print(f"\t Before removing outlier we had {data.shape[0]} rows")

    z = zscore(df[var])
    df_filtered = df[(z > -3) & (z < 3)]
    
    print(f"\t After removing outlier we have {df_filtered.shape[0]} rows")
    
    # 3. Test the normality of the all observation
    print("Re testing the normality:")
    result, conclusion = test_stats_normality(df_filtered, var, mu, sigma, alpha)
    return (result, conclusion)

def compare_distribution_with_normal_distribution(df, var, mu, sigma, n, bins=20):
    # variable 
    data = df[var]
    n_var = n 
    
    # Plot 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))

    # 1.1 plot histogramme empirique 
    axes[0].hist(data, bins=bins, density=True, alpha=0.6, edgecolor='b')
    
    # 1.1 plot normal distribution 
    x = np.linspace(min(data), max(data), 1000)
    y = norm.pdf(x, loc=mu, scale=sigma)
    axes[0].plot(x,y, color='r', label='Loi Normal Théorique')
    
    axes[0].set_title(f"Histogramme de '{var}' et courbe de la loi normale")
    axes[0].set_xlabel(var)
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True)

    # 2.1. eCDF
    x_var = np.sort(data)
    y_var = np.arange(1, n_var+1)/n_var
    axes[1].plot(x_var, y_var, marker='.', linestyle='none', alpha=0.5, label=f'eCDF {var}')


    # 2.2 CDF
    x_norm = np.linspace(min(data), max(data), n_var)
    y_norm  = norm.cdf(x_norm, loc=mu, scale=sigma)
    axes[1].plot(x_norm,y_norm, marker='.', color='r', linestyle='none',  alpha=0.5, label='Loi Normal Théorique')
    
    axes[1].set_title(f"Fonction de répartition empirique de {var} et théorique de la loi normale")
    axes[1].set_xlabel(f'{var}')
    axes[1].set_ylabel('CDF')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Comparaison avec la loi normale")
    plt.show()
    return 

def test_normality(df, var, alpha=0.05, bins=20, remove_outlier = False):
    from scipy.stats import shapiro, ks_2samp, kstest
    # 1. load data 
    print(var)
    data = df[var]
    data = data.dropna()

    # 2. Calculate the mean and standard deviation
    mu = data.mean()
    sigma = np.std(data)
    n_var = len(data)
    
    # 3. Plot 
    compare_distribution_with_normal_distribution(df, var, mu, sigma, n_var, bins=20)

    # 4. Test the normality
    if remove_outlier: 
        result, conclusion = compare_normality_pre_post_outlier_removal(df, var , mu, sigma, alpha)
    else: 
        result, conclusion = test_stats_normality(df, var,  mu, sigma, alpha)
    print(result)
    print(conclusion)
    return 


def test_normality_def_old(df, var, bins=20):
    from scipy.stats import shapiro, ks_2samp, kstest
    # 1. load data 
    print(var)
    data = df[var]
    data = data.dropna()

    # 2. Calculate the mean and standard deviation
    mu = data.mean()
    sigma = np.std(data)
    n_var = len(data)
    
    # 3. Plot 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))

    # 3.1 plot histogramme empirique 
    axes[0].hist(data, bins=bins, density=True, alpha=0.6, edgecolor='b')
    
    # 3.1 plot normal distribution 
    x = np.linspace(min(data), max(data), 1000)
    y = norm.pdf(x, loc=mu, scale=sigma)
    axes[0].plot(x,y, color='r', label='Loi Normal Théorique')
    
    axes[0].set_title(f"Histogramme de '{var}' et courbe de la loi normale")
    axes[0].set_xlabel(var)
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True)


    # 3.1. eCDF
    x_var = np.sort(data)
    y_var = np.arange(1, n_var+1)/n_var
    axes[1].plot(x_var, y_var, marker='.', linestyle='none', alpha=0.5, label=f'eCDF {var}')


    # 3.2 CDF
    x_norm = np.linspace(min(data), max(data), n_var)
    y_norm  = norm.cdf(x_norm, loc=mu, scale=sigma)
    axes[1].plot(x_norm,y_norm, marker='.', color='r', linestyle='none',  alpha=0.5, label='Loi Normal Théorique')
    
    axes[1].set_title(f"Fonction de répartition empirique du {var} et théorique de la loi normale")
    axes[1].set_xlabel(f'{var}')
    axes[1].set_ylabel('CDF')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Comparaison avec la loi normale")
    plt.show()

    # 4. Test the normality
    compare_normality_pre_post_outlier_removal(df, var , mu, sigma)

# ** Statistical Tests ** --------------------------------------------------------------------------------------------------
def display_distrib_and_boxplot_of_2_groups_var(df, variable, var_group): 
    # --- Créer la figure avec 3 axes ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    group_val = sorted(df[var_group].unique())
    
    
    # --- Filtrer les deux classes ---
    df_g1 = df[df[var_group] == group_val[0]]
    df_g2 = df[df[var_group] == group_val[1]]
    
    # --- Histogrammes par groupes  ---
    sns.histplot(data=df_g1, x=variable, ax=axes[0], kde=True, bins=20)
    axes[0].set_title(f"Distribution of: '{var_group}'= {group_val[0]}")
    
    sns.histplot(data=df_g2, x=variable, ax=axes[1], kde=True, bins=20)
    axes[1].set_title(f"Distribution of: '{var_group}'= {group_val[1]}")
    
    # --- Synchroniser les echelles X, Y  ---
    max_y = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, max_y)
    axes[1].set_ylim(0, max_y)
    
    common_xlim = (
        min(axes[0].get_xlim()[0], axes[1].get_xlim()[0]),
        max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    )
    axes[0].set_xlim(common_xlim)
    axes[1].set_xlim(common_xlim)
    
    # --- Boxplot ---
    sns.boxplot(data=df, hue=var_group, y=variable, ax=axes[2])
    axes[2].set_title("Boxplot by group")
    
    plt.suptitle(f"Distribution {variable} by {var_group} + boxplot") 
    plt.tight_layout()
    plt.show()
    return

def test_moyenne_2_groupes_def(df, variable, var_group, group_value = [0,1], alpha = 0.05): 
    # Vérifions si bmi suit le lois normale par groupes. 
    g1_val, g2_val = group_value
    g1 = df[variable][df[var_group]==g1_val]
    g2 = df[variable][df[var_group]==g2_val]

    print(f"Test if there is a significant differences of the average {variable}, between the {var_group}, at the significant level alpha = {alpha}")

    # Plot distribution 
    display_distrib_and_boxplot_of_2_groups_var(df, variable, var_group)
    
    # Test de shapiro 
    alpha = 0.05
    shapiro_stat_g1, shapiro_p_value_g1  = shapiro(g1)
    shapiro_stat_g2, shapiro_p_value_g2  = shapiro(g2)
    
    # Interprétation 
    print("\n\t**Test de shapiro**")
    if shapiro_p_value_g1 < alpha: 
        print(f"The p_value for '{var_group} = {g1_val}' is {shapiro_p_value_g1} < {alpha}, so we reject the null hypothesis, and considere that distribution is not normal")
    else: 
        print(f"The p_value for '{var_group} = {g1_val}' is {shapiro_p_value_g1:.4f} > {alpha}, so we fail to reject the null hypothesis, and considerate the distribution as normal")
    if shapiro_p_value_g2 < alpha:
        print(f"The p_value for '{var_group} = {g2_val}' is {shapiro_p_value_g2:} < {alpha}, so we reject the null hypothesis, and considere that distribution is not normal")
    else: 
        print(f"The p_value for '{var_group} = {g2_val}' is {shapiro_p_value_g2:.4f} > {alpha}, so we fail to reject the null hypothesis, and considerate the distribution as normal")
    
    # Test d'égalité des variance 
    print("\n\t**Test de levene**")
    levene_stat, levene_p_value = levene(g1, g2)
    if levene_p_value < alpha: 
        print("Les variance ne sont pas égales")
    else:
        print("Les variance sont égales")

    # Test des moyennes: 
    if shapiro_p_value_g1 > alpha and shapiro_p_value_g2 > alpha and levene_p_value > alpha:
        print("\n\t** T-test **")
        t_stat, t_p_value = ttest_ind(g1, g2, equal_var=True)
        print(f"stat ={t_stat}, p_value = {t_p_value}")
        if t_p_value < alpha: 
            print(f"The p_value < {alpha}, we rejet the null hypothesis, and the conclude that there is a significant difference between each '{var_group}' groups")
        else: 
            print(f"The p_value: {t_p_value:.4f} > {alpha}, we fail to rejet the null hypothesis, and the conclude that there is no significant difference between each '{var_group}' groups")
    
    elif (shapiro_p_value_g1 > alpha and shapiro_p_value_g2 > alpha) and levene_p_value < alpha: 
        print("\n\t** Welch test **")
        welch_stat, welch_p_value = ttest_ind(g1, g2, equal_var=False)
        print(f"stat ={welch_stat}, p_value = {welch_p_value}")
        if welch_p_value < alpha: 
            print(f"The p_value < {alpha}, we rejet the null hypothesis, and the conclude that there is a significant difference between each '{var_group}' groups")
        else: 
            print(f"The p_value: {welch_p_value:.4f} > {alpha}, we fail to rejet the null hypothesis, and the conclude that there is no significant difference between each '{var_group}' groups")
    
        
    elif (shapiro_p_value_g1 < alpha or shapiro_p_value_g2 < alpha): 
        print("\n\t** Mann Whitney U**")
        print(f"As the {variable} is not normaly distributed within the group, we will perform the Mann Whitney U test, and we also perform Welch’s t-test (since n > 30), for comparison.")
        # Welch
        welch_stat, welch_p_value =  ttest_ind(g1, g2, equal_var=False)
        
        # Mann-Whitney
        mwu_stat, mwu_p_value =  mannwhitneyu(g1, g2, alternative='two-sided')
        
        result_dict = {'test': ['Welch','Mann Whitney U'], 'statistic': [welch_stat, mwu_stat], 'p_value': [welch_p_value, mwu_p_value]}
        result = pd.DataFrame(result_dict)
        print(result, "\n")

        if mwu_p_value < alpha: 
            print(f"The p_value < {alpha}, we rejet the nullhypothesis, and the conclude that there is a significant difference between each '{var_group}' groups")
        else: 
            print(f"The p_value: {mwu_p_value:.4f} > {alpha}, we fail to rejet the nullhypothesis, and the conclude that there is no significant difference between each '{var_group}' groups")

    return

def check_normality_by_group(df, variable, var_group, alpha = 0.01):
    # Vérifions la normalité par groupe
    sns.displot(data=df, x=variable, col=var_group, kde=True)
    plt.suptitle(f"Distribution of {variable} by {var_group}")
    plt.tight_layout()
    plt.show()
    
    # Test de shapiro 
    stat_list, pval_list = [], [] 
    group_value = df[var_group].cat.categories.tolist()
    for i in group_value: 
        stat, p_value = shapiro(df[df[var_group] == i][variable])
        stat_list.append(stat)
        pval_list.append(p_value)
    result = {'groups':group_value, 'statistic':stat_list, 'p value': pval_list}
    df_result = pd.DataFrame(result).round(4)
    
    print("** Shapiro Test**\n")
    print(df_result, "\n")
    if (df_result['p value'] > alpha).all(): 
        print(f"For all the groups the p_value > alpha = {alpha}, so the distribution {variable} is normal within the groups")
    else: 
        print(f"For at least one group the p_value < alpha = {alpha}, so we reject the null hypothesis and the {variable} distribution is not normal within the groups")
    return df_result

def check_levene_by_group(df, variable, var_group, alpha = 0.05):
    print("\n** Test equality of variance - Levene test**\n")
    # test de variances
    group_list = []
    for cat in df[var_group].cat.categories: 
        group = df[df[var_group] == cat][variable]
        group_list.append(group)
    result = levene(*group_list)
    if result.pvalue > alpha: 
        print(f"p_value > alpha = {alpha}, so fail to reject the null hypothesis and the variances of {variable} within a group are equals")
    else: 
        print(f"p_value < alpha = {alpha}, so we reject the null hypothesis and the variances of {variable} within the groups are not equals")
    return result 

def eda_for_anova(df, variable, var_group):
    # Plot boxplot 
    sns.boxplot(data=df, y=variable, hue=var_group)
    plt.show()
    
    # Calcule moyenne  + d'écart type par groupe
    mean_std_by_group = pd.DataFrame(df.groupby(var_group)[variable].agg(['mean','std'])).reset_index().round(1)
    return mean_std_by_group


# ** Multivariate Analysis ** --------------------------------------------------------------------------------------------------

def select_highest_correlation(corr_matrix): 
    # Select the highest correlation (> 0.70 or <-0.70)
    mask = np.triu(np.ones(corr_matrix.shape)).astype(bool) # Filter to select the upper triangle of the matrix
    corr_matrix_melt = corr_matrix.mask(mask).stack().reset_index()
    corr_matrix_melt.columns =['variable_1','variable_2','correlation']
    corr_matrix_melt['abs_correlation'] = abs(corr_matrix_melt['correlation'])
    result = corr_matrix_melt[abs(corr_matrix_melt['correlation']) > 0.70]
    return result

def vif_analysis(df, variables, add_const = True): 
    # Select predictors variables 
    X = df[variables]
    print("df shape: ", X.shape)
    print("df initial variables: ", list(X.columns))
    
    # transform data
    X = pd.get_dummies(X, drop_first=True)
    bool_var = X.select_dtypes(include=['bool']).columns
    X[bool_var] = X[bool_var].astype(int)
    
    X = X.dropna()
    # Add constant (if needed)
    if add_const:
        X = add_constant(X)
    print("Selected variable for the vif analysis ", list(X.columns))

    # Check VIF can be performed correctly 
    print("\nSanity check: ")
    print("\t Null values: ", X.isnull().sum().sum())  # Check nan values
    print("\t Nan values: ", X.isna().sum().sum())
    print("\t infinite values: ", np.isinf(X).sum().sum())  # Check infinite values
    print(f"\t Constant columns: {X.nunique() [ X.nunique() < 2].index.values}") # Check the constant columns
    print("\t Boolean columns:", X.select_dtypes(include=['bool']).columns.values) # Check if there is bool columns 

    # Calculate the vif for each feature
    data_vif = pd.DataFrame()
    data_vif['Feature'] = X.columns
    data_vif['Value'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return data_vif

# ** Statistical Modelisation ** -------------------------------------------------------------------------------------------

def evaluate_ols_regression_model(my_model, y_train, X_test, y_test):
    # Metric of the train set 
    print("***Metric of the train set***")
    y_pred_train = my_model.fittedvalues
    assert len(y_pred_train) == len(y_train)

    r_squared_train = my_model.rsquared
    r2_train = r2_score(y_train, y_pred_train)
    assert r_squared_train == r2_train
    
    print(f"Mean of squared residuals: {my_model.mse_resid:.4f}")
    print(f"r_squared train = {r2_train:.4f}")
    print(f"MAPE: {mape(y_train, y_pred_train):.4f}")
    print(f"RMSE train = {mean_squared_error(y_train, y_pred_train)**(1/2):.4f}")
    
    # Metric of the test set
    print("\n***Metric of the test set***")
    y_pred = my_model.predict(X_test)
    assert len(y_pred) == len(y_test)
    
    mape_test = mape(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse_test = mean_squared_error(y_test, y_pred)**(1/2)
    
    print(f"r_squared test = {r2:.4f}")
    print(f"MAPE test = {mape(y_test, y_pred):.4f}")
    print(f"RMSE test = {rmse_test:.4f}")

    # Conclusion
    print(f"\nThe coefficient of determination R2 varies from {r2_train:.3f} during the training to {r2:.3f} with the test set")
    return y_pred_train, y_pred

# ** Machine Learning Models ** --------------------------------------------------------------------------------------------
def update_model_results(df, model_name, rmse_train, cv_mean_rmse, cv_std_rmse):
    """
    Create a dataframe of each model RMSE performance, in order to compare them. 
    """
    if model_name in df['model'].unique():
        df.loc[df['model'] == model_name,["RMSE Train","CV Mean RMSE", "CV Std RMSE"]] = [rmse_train, cv_mean_rmse, cv_std_rmse]
    else:
        new_row = {
            "model": model_name,
            "RMSE Train": rmse_train,
            "CV Mean RMSE": cv_mean_rmse, 
            "CV Std RMSE": cv_std_rmse
        }
        df = pd.concat([df, pd.DataFrame([new_row])], axis=0, ignore_index=True)
    return df.round(3)

def fit_and_evaluate_model_with_cv(name, model, X_train, y_train, kf): 
    """
    Fit a model on the training set and evaluate it on the same set. 
    Then, performe a cross validation evluation on the same initial train set.
    Display the same metrics for both: r2, mae, rmse
    + cv rmse score for the training and test folds
    And return only rmse metrcis: rmse perform on the entire training set, mean rsme of test fold of the CV + standard deviation
    """
    scorings = {'r2':'r2', 'neg_mae':'neg_mean_absolute_error', 'neg_rmse':'neg_root_mean_squared_error'}
    # Evaluation on en entire training set 
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    r2_score_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)

    # Cross Validation on the training set 
    cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring= scorings, return_train_score=True)
    cv_r2_test = cv_results['test_r2']
    cv_mae_test = -cv_results['test_neg_mae']
    cv_rmse_test =  -cv_results['test_neg_rmse']
    cv_rmse_train =  -cv_results['train_neg_rmse']

    mean_cv_rmse_test = cv_rmse_test.mean()
    std_cv_rmse_test = cv_rmse_test.std()
        
    print(f"{name.capitalize()}: Fitting and Evaluation on the entire train set and with Cross validation (from the trainset)")
    print(f"R2 train = {r2_score_train:.3f}     | MAE Train  = {mae_train:.3f}    | RMSE Train = {rmse_train:.3f}")
    print(f"R2 mean (CV) = {cv_r2_test.mean():.3f} | MAE mean (CV) = {cv_mae_test.mean():.3f} | RMSE mean (CV)  = {mean_cv_rmse_test:.3f}")
    print(f"RMSE CV score:")
    print(f" - cv rmse score on training folds: {cv_rmse_train} | std: {cv_rmse_train.std():.3f} \n - cv rmse score test folds: {cv_rmse_test} | std: {std_cv_rmse_test:.3f}")
    return rmse_train, mean_cv_rmse_test, std_cv_rmse_test
            