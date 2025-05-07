import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.isotonic import isotonic_regression

from tqdm import tqdm
import timeit
import os

from isodisreg import idr       # to compute isotonic regression fit
import urocc                    # to compute CPA


save_plots = 'plots/'

# define possible error measures ----------------------------------------------------

def rmse(pred, y):
    # root mean squared error
    return np.sqrt(np.mean((pred - y)**2))

def acc(pred, y):
    # anomaly correlation coefficient corresponds simply to the Pearson correlation
    # numpy just evaluates correlation matrix, i.e., correlation coefficient is at [0,1] or at [1,0]
    return np.corrcoef(pred, y)[0, 1]

def mae(pred, y):
    # mean absolute loss
    return np.mean(np.abs(pred - y))

def ql90(pred, y):
    # quantile loss at level 0.9
    return np.mean(((pred > y) - 0.9) * (pred - y))

def cpa(pred, y):
    # coefficient of predictive ability : average AUC values for all possible binarized problems
    return urocc.cpa(y, pred)

def auc(pred, y):
    # calculate AUC
    return metrics.roc_auc_score(y, pred)

def auc_plus(pred, y):
    # calculate AUC after replacing ROC curve by its concave hull
    pair_array = np.array(list(zip(pred, -1 * y)), dtype=[('x', float), ('-y', float)])
    # sort x increasing, and in case of ties use y to determine decreasing order
    order = np.argsort(pair_array, order=('x', '-y'))
    recalibrate = isotonic_regression(y[order])
    return metrics.roc_auc_score(y[order], recalibrate)

def pc(pred, y):
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    # "predict" the estimated cdfs again, so that we can use the crps function
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))
    mean_crps = np.mean(prob_pred.crps(y))
    return mean_crps

def my_crps(x, cum_weights, y):
    weights = cum_weights - np.hstack((np.zeros(((y.size, 1))), cum_weights[:, :-1]))
    # the formula is simply extracted from idr predict crps function
    return 2 * np.sum(weights * (np.array((y < x)) - cum_weights + 0.5 * weights) * np.array(x - y), axis=1)

def pc_time(pred, y):
    time1 = timeit.default_timer()
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    time2 = timeit.default_timer()
    mean_crps = np.mean(my_crps(np.reshape(fitted_idr.thresholds, (1, -1)), # reshape to make it braodcast-able
                                fitted_idr.ecdf,
                                np.concatenate(fitted_idr.y.values).reshape((-1, 1))))
    time3 = timeit.default_timer()
    print(mean_crps)
    return np.array([time2 - time1, time3 - time2])

def pcs(pred, y):
    # crps of the climatological forecast
    pc_ref = np.mean(np.abs(np.tile(y, (len(y), 1)) - np.tile(y, (len(y), 1)).transpose())) / 2

    return (pc_ref - pc(pred, y)) / pc_ref


# simulation routines ----------------------------------------------------------------

def get_data(n = 1000, seed = 1):
    np.random.seed(seed)
    w = np.random.uniform(low=0.0, high=10.0, size=n)

    my_shape = np.sqrt(w)
    my_scale = np.minimum(np.maximum(w, 1), 6)
    y = np.random.gamma(shape=my_shape, scale=my_scale, size=n)

    return pd.DataFrame({'y': y,
                         'f1': w,
                         'f2': my_shape * my_scale,
                         'f3': gamma.ppf(0.5, my_shape, scale=my_scale),
                         'f4': gamma.ppf(0.9, my_shape, scale=my_scale)})


def run_simulation_example_1(n=1000, square_y=False):
    pred_data = get_data(n=n)
    plot_data(pred_data)

    y = pred_data['y']
    add_name = ''
    if square_y:
        y = pred_data['y']**2
        add_name = '_squared'

    loss_fcts = {'RMSE': rmse, 'MAE': mae, 'QL90': ql90, 'PC': pc, 'ACC': acc, 'CPA': cpa, 'PCS': pcs}

    fcsts = pred_data.columns[pred_data.columns != 'y']
    loss_vals = np.zeros((len(fcsts), len(loss_fcts)))

    for i in range(len(fcsts)):
        for j, loss in enumerate(loss_fcts.values()):
            loss_vals[i, j] = loss(pred_data[fcsts[i]], y)

    res = pd.DataFrame(loss_vals, columns=list(loss_fcts.keys()))
    res.insert(0, 'Model', fcsts)
    res.to_csv(os.path.join(save_plots, 'loss_values' + add_name + '.csv'))
    print(res)


def run_simulation_example_2(thresh_list=[10], add_name=''):
    df_all = pd.DataFrame()

    pred_data = get_data(n=10000)
    loss_fcts = {'PC': pc, 'PCS': pcs, 'AUC': auc, 'AUC+': auc_plus}
    fcsts = pred_data.columns[pred_data.columns != 'y']
    loss_vals = np.zeros((len(fcsts), len(loss_fcts)))

    for t in tqdm(thresh_list):
        for i in range(len(fcsts)):
            for j, loss in enumerate(loss_fcts.values()):
                loss_vals[i, j] = loss(pred_data[fcsts[i]], 1 * (pred_data['y'] >= t))

        res = pd.DataFrame(loss_vals, columns=list(loss_fcts.keys()))
        res.insert(0, 'Threshold', np.repeat(t, res.shape[0]))
        res.insert(0, 'Model', fcsts)

        df_all = pd.concat((df_all, res))

    df_all.to_csv(os.path.join(save_plots, 'loss_values_bin' + add_name + '.csv'))



# print and plot  ----------------------------------------------------------------

def print_results():
    column_order = ['RMSE', 'MAE', 'QL90', 'PC', 'ACC', 'CPA', 'PCS']

    t = pd.read_csv(os.path.join(save_plots, 'loss_values.csv'))
    print(t.loc[:, column_order].round({'RMSE': 2, 'MAE': 2, 'QL90': 2, 'PC': 2, 'ACC': 3, 'CPA': 3, 'PCS': 3}))
    t = pd.read_csv(os.path.join(save_plots, 'loss_values_squared.csv'))
    print(t.loc[:, column_order].round({'RMSE': 0, 'MAE': 0, 'QL90': 0, 'PC': 0, 'ACC': 3, 'CPA': 3, 'PCS': 3}))


def plot_data(df):
    plt.figure()
    plt.title('Pairwise Scatter Plot')
    pd.plotting.scatter_matrix(df, figsize=(10, 10))
    plt.savefig(os.path.join(save_plots, 'pairwise_plot.png'))
    plt.close()

    df_sorted = df.sort_values(by='f1')
    plt.figure()
    plt.scatter(df_sorted['f1'], df_sorted['y'], label='Y', c='black', s=5)
    plt.plot(df_sorted['f1'], df_sorted['f2'], label='Cond. mean')
    plt.plot(df_sorted['f1'], df_sorted['f3'], label='Cond. median')
    plt.plot(df_sorted['f1'], df_sorted['f4'], label='Cond. 90%-quantile')
    plt.legend()
    plt.savefig(os.path.join(save_plots, 'scatter_plot.png'))
    plt.close()


def plot_thresh_graph():
    df = pd.read_csv(os.path.join(save_plots, 'loss_values_bin_graph.csv'), index_col=0)
    # as all forecaster have same statistic filter one of them
    df = df.loc[df['Model'] == 'f1']
    stat_list = ['PC', 'PCS', 'AUC']    # take 'AUC+' out
    df_long = pd.melt(df, id_vars='Threshold', value_vars=stat_list, var_name='Stat')

    sns.set_theme(style='whitegrid')
    g = sns.lineplot(data=df_long, x='Threshold', y='value', hue='Stat', palette='husl')
    g.figure.set_size_inches(6.5, 4.5)
    g.set(xlabel='Threshold c', ylabel='', title='')
    plt.legend(title='')
    g.get_figure().savefig(os.path.join(save_plots, 'stat_by_threshold.png'))


def analyze_runtime():
    n_vals = 10**np.arange(2,5)

    time_measures = np.zeros((4 * len(n_vals), 2))

    for s, n in enumerate(n_vals):
        pred_data = get_data(n=n)
        fcsts = pred_data.columns[pred_data.columns != 'y']
        for i in range(len(fcsts)):
            time_measures[s * 4 + i] = pc_time(pred_data[fcsts[i]], pred_data['y'])

    res = pd.DataFrame(time_measures, columns=['IDR Fit', 'CRPS'])
    res.insert(0, 'n', np.repeat(n_vals, len(fcsts)))
    res.insert(0, 'Model', np.tile(fcsts, len(n_vals)))
    res.to_csv(os.path.join(save_plots, 'runtime.csv'))
    print(res)


def plot_runtime():
    df = pd.read_csv(os.path.join(save_plots, 'runtime.csv'), index_col=0)
    df_plot = df.groupby(['n', 'Model'])[['IDR Fit', 'CRPS']].mean().reset_index()
    df_long = pd.melt(df_plot, id_vars=['n', 'Model'], value_vars=['IDR Fit', 'CRPS'], var_name='Routine')

    sns.set_theme(style='whitegrid')
    g = sns.lineplot(data=df_long, x='n', y='value', hue='Routine', style='Model', palette='husl')
    g.figure.set_size_inches(6.5, 4.5)
    g.set(xlabel='Input size [n]', ylabel='Runtime [s]', title='')
    g.set(xscale='log', yscale='log')
    plt.legend(title='')
    plt.tight_layout()
    g.get_figure().savefig(os.path.join(save_plots, 'runtime.png'))


if __name__ == '__main__':
    if not os.path.exists(save_plots):
        print(f'Please create directory {save_plots}')
    else:
        # run_simulation_example_1(n=10000, square_y=True)
        # run_simulation_example_2(thresh_list=np.linspace(1, 40, 100), add_name='_graph')
        # plot_thresh_graph()
        # print_results()
        # analyze_runtime()
        plot_runtime()
