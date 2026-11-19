# Run from the folder that contains data_cleaned, data_original and stage2_model: python stage2_model/stage2_model.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from math import sqrt
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

os.makedirs('stage2_model/outputs', exist_ok=True)

sp500 = pd.read_csv('data_cleaned/cleaned_data.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'], format='%Y-%m-%d')
sp500 = sp500.sort_values('Date')
sp500['Month'] = sp500['Date'].dt.to_period('M')

df = sp500.groupby('Month').agg(Open=('Open', 'first'), Close=('Close', 'last')).reset_index()
df['Return_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100

print('Months of S&P 500 data:', len(df), '(', df['Month'].min(), 'to', df['Month'].max(), ')')

ffr = pd.read_csv('data_original/ffr_original.csv')
ffr['Month'] = pd.to_datetime(ffr['observation_date']).dt.to_period('M')
ffr = ffr.sort_values('Month')

ffr['FFR_change'] = ffr['FEDFUNDS'].diff()

# shift moves FFR down by 1-3 months, so each month only sees FFR from before it
ffr['FFR_lag1'] = ffr['FEDFUNDS'].shift(1)
ffr['FFR_lag2'] = ffr['FEDFUNDS'].shift(2)
ffr['FFR_lag3'] = ffr['FEDFUNDS'].shift(3)
ffr['FFR_change_lag1'] = ffr['FFR_change'].shift(1)
ffr['FFR_change_lag2'] = ffr['FFR_change'].shift(2)
ffr['FFR_change_lag3'] = ffr['FFR_change'].shift(3)
ffr['FFR_change_3m'] = ffr['FFR_change_lag1'] + ffr['FFR_change_lag2'] + ffr['FFR_change_lag3']

ffr['FFR_direction_3m'] = 0
ffr.loc[ffr['FFR_change_3m'] > 0.10, 'FFR_direction_3m'] = 1
ffr.loc[ffr['FFR_change_3m'] < -0.10, 'FFR_direction_3m'] = -1

input_names = ['FFR_lag1', 'FFR_lag2', 'FFR_lag3',
               'FFR_change_lag1', 'FFR_change_lag2', 'FFR_change_lag3',
               'FFR_change_3m', 'FFR_direction_3m']
target_name = 'Return_Pct'

df = df.merge(ffr[['Month'] + input_names], on='Month', how='left')
if df[input_names].isna().any().any():
    print('Warning: some months have no FFR inputs')
df.to_csv('stage2_model/outputs/model_data.csv', index=False)

# split by year instead of train_test_split, so the model never learns from the future
train = df[df['Month'].dt.year <= 2018]
test = df[df['Month'].dt.year >= 2019]

X_train = train[input_names]
y_train = train[target_name]
X_test = test[input_names]
y_test = test[target_name]

print('Training months:', len(train), '| Testing months:', len(test))
print('Split: {:.0%} train / {:.0%} test'.format(len(train) / len(df), len(test) / len(df)))

tscv = TimeSeriesSplit(n_splits=5)
scores = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']

baseline_value = y_train.mean()
baseline_mae, baseline_rmse, baseline_r2 = [], [], []
for train_index, val_index in tscv.split(X_train):
    guess = [y_train.iloc[train_index].mean()] * len(val_index)
    actual = y_train.iloc[val_index]
    baseline_mae.append(metrics.mean_absolute_error(actual, guess))
    baseline_rmse.append(sqrt(metrics.mean_squared_error(actual, guess)))
    baseline_r2.append(metrics.r2_score(actual, guess))

lasso_grid = GridSearchCV(
    make_pipeline(StandardScaler(), linear_model.Lasso(max_iter=50000)),
    {'lasso__alpha': np.logspace(-3, 1, 50)},
    cv=tscv, scoring=scores, refit='neg_mean_absolute_error')
lasso_grid.fit(X_train, y_train)
lasso = lasso_grid.best_estimator_
lasso_coef = lasso.named_steps['lasso'].coef_

gb_grid = GridSearchCV(
    ensemble.GradientBoostingRegressor(subsample=0.8, random_state=42),
    {'n_estimators': [50, 100, 200],
     'max_depth': [1, 2, 3],
     'learning_rate': [0.01, 0.05]},
    cv=tscv, scoring=scores, refit='neg_mean_absolute_error')
gb_grid.fit(X_train, y_train)
gb = gb_grid.best_estimator_

def train_cv_scores(grid):
    best = grid.best_index_
    mae = -grid.cv_results_['mean_test_neg_mean_absolute_error'][best]
    rmse = -grid.cv_results_['mean_test_neg_root_mean_squared_error'][best]
    r2 = grid.cv_results_['mean_test_r2'][best]
    return mae, rmse, r2

lasso_cv = train_cv_scores(lasso_grid)
gb_cv = train_cv_scores(gb_grid)
base_cv = (np.mean(baseline_mae), np.mean(baseline_rmse), np.mean(baseline_r2))

train_cv_results = pd.DataFrame(
    [base_cv, lasso_cv, gb_cv],
    index=['Baseline (average month)', 'Lasso', 'Gradient Boosting'],
    columns=['Train_CV_MAE', 'Train_CV_RMSE', 'Train_CV_R2'])
train_cv_results.to_csv('stage2_model/outputs/training_cv_results.csv')

print('\n----- Training (cross-validated inside 2001-2018) -----')
print(train_cv_results.round(4).to_string())
print('Lasso best alpha:', round(lasso_grid.best_params_['lasso__alpha'], 4))
print('Gradient Boosting best settings:', gb_grid.best_params_)

if lasso_cv[0] <= gb_cv[0]:
    chosen = 'Lasso'
else:
    chosen = 'Gradient Boosting'
print('Model chosen on training:', chosen)

y_pred_base = [baseline_value] * len(y_test)
y_pred_lasso = lasso.predict(X_test)
y_pred_gb = gb.predict(X_test)

def test_scores(y_true, y_pred):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = sqrt(metrics.mean_squared_error(y_true, y_pred))
    r2 = metrics.r2_score(y_true, y_pred)
    direction = (np.sign(y_pred) == np.sign(y_true)).mean() * 100
    return mae, rmse, r2, direction

results = pd.DataFrame(
    [test_scores(y_test, y_pred_base),
     test_scores(y_test, y_pred_lasso),
     test_scores(y_test, y_pred_gb)],
    index=['Baseline (average month)', 'Lasso', 'Gradient Boosting'],
    columns=['MAE', 'RMSE', 'R2', 'Direction_Accuracy_%'])
results.to_csv('stage2_model/outputs/model_results.csv')

print('\n----- Test (2019-2023) -----')
print(results.round(4).to_string())
print('Final estimate for', chosen, '- MAE:', round(results.loc[chosen, 'MAE'], 4),
      '| RMSE:', round(results.loc[chosen, 'RMSE'], 4))
print('Share of test months that went up: {:.1f}%'.format((y_test > 0).mean() * 100))

actual_up = y_test > 0
direction_rows = []
for name, pred in [('Baseline (average month)', y_pred_base), ('Lasso', y_pred_lasso), ('Gradient Boosting', y_pred_gb)]:
    pred_up = np.array(pred) > 0
    direction_rows.append([metrics.accuracy_score(actual_up, pred_up),
                           metrics.balanced_accuracy_score(actual_up, pred_up),
                           metrics.precision_score(actual_up, pred_up),
                           metrics.recall_score(actual_up, pred_up),
                           metrics.f1_score(actual_up, pred_up)])
direction_results = pd.DataFrame(direction_rows,
                                 index=['Baseline (average month)', 'Lasso', 'Gradient Boosting'],
                                 columns=['Accuracy', 'Balanced_accuracy', 'Precision_up', 'Recall_up', 'F1_up'])
direction_results.to_csv('stage2_model/outputs/direction_results.csv')
print('\n----- Up/down scores on the test months -----')
print(direction_results.round(4).to_string())

chosen_pred = y_pred_gb if chosen == 'Gradient Boosting' else y_pred_lasso
confusion = metrics.confusion_matrix(actual_up, np.array(chosen_pred) > 0)
print('Confusion matrix for', chosen, '(rows = actual down/up, columns = predicted down/up):')
print(confusion)

train_results = pd.DataFrame(
    [test_scores(y_train, [baseline_value] * len(y_train)),
     test_scores(y_train, lasso.predict(X_train)),
     test_scores(y_train, gb.predict(X_train))],
    index=['Baseline (average month)', 'Lasso', 'Gradient Boosting'],
    columns=['MAE', 'RMSE', 'R2', 'Direction_Accuracy_%'])
train_results.to_csv('stage2_model/outputs/model_results_train.csv')
print('\n----- Training months (overfitting check) -----')
print(train_results.round(4).to_string())

pd.DataFrame({'Month': test['Month'].astype(str), 'Actual': y_test,
              'Baseline': y_pred_base, 'Lasso': y_pred_lasso,
              'Gradient_Boosting': y_pred_gb}).to_csv(
    'stage2_model/outputs/test_predictions.csv', index=False)

print('\nLasso coefficients (0 = input dropped):')
for name, c in zip(input_names, lasso_coef):
    print('  {:<18} {:.4f}'.format(name, c))
print('Gradient Boosting feature importances:')
for name, imp in zip(input_names, gb.feature_importances_):
    print('  {:<18} {:.4f}'.format(name, imp))

pd.DataFrame({'Lasso_coefficient': lasso_coef,
              'GB_importance': gb.feature_importances_},
             index=input_names).to_csv('stage2_model/outputs/what_models_learned.csv')

df['Set'] = np.where(df['Month'].dt.year <= 2018, 'Train 2001-2018', 'Test 2019-2023')
df['FFR_trend'] = df['FFR_direction_3m'].map({-1: 'Falling', 0: 'Same', 1: 'Rising'})
by_direction = df.groupby(['Set', 'FFR_trend'])['Return_Pct'].agg(['count', 'mean', 'median', 'std'])
by_direction.to_csv('stage2_model/outputs/return_by_ffr_direction.csv')
print('\nNext-month return (%) after FFR rose / stayed / fell over 3 months:')
print(by_direction.round(3))

sample = ffr[ffr['Month'] == pd.Period('2024-01', 'M')][input_names]
print('\n----- Sample case: predicting January 2024 -----')
print('FFR Oct, Nov, Dec 2023:', sample['FFR_lag3'].iloc[0], sample['FFR_lag2'].iloc[0], sample['FFR_lag1'].iloc[0])
print('Lasso predicted return: {:.2f}%'.format(lasso.predict(sample)[0]))
print('Gradient Boosting predicted return: {:.2f}%'.format(gb.predict(sample)[0]))
print('Baseline guess: {:.2f}%'.format(baseline_value))

robust = []
for split_year in range(2014, 2020):
    tr = df[df['Month'].dt.year <= split_year]
    te = df[df['Month'].dt.year > split_year]
    l_model = GridSearchCV(make_pipeline(StandardScaler(), linear_model.Lasso(max_iter=50000)),
                           {'lasso__alpha': np.logspace(-3, 1, 50)},
                           cv=tscv, scoring='neg_mean_absolute_error').fit(tr[input_names], tr[target_name])
    g_model = GridSearchCV(ensemble.GradientBoostingRegressor(subsample=0.8, random_state=42),
                           {'n_estimators': [50, 100, 200], 'max_depth': [1, 2, 3],
                            'learning_rate': [0.01, 0.05]},
                           cv=tscv, scoring='neg_mean_absolute_error').fit(tr[input_names], tr[target_name])
    robust.append([split_year, len(tr), len(te),
                   metrics.mean_absolute_error(te[target_name], [tr[target_name].mean()] * len(te)),
                   metrics.mean_absolute_error(te[target_name], l_model.predict(te[input_names])),
                   metrics.mean_absolute_error(te[target_name], g_model.predict(te[input_names]))])
robust = pd.DataFrame(robust, columns=['Train_up_to', 'Train_months', 'Test_months',
                                       'Baseline_MAE', 'Lasso_MAE', 'GB_MAE'])
robust.to_csv('stage2_model/outputs/robustness_split_years.csv', index=False)
print('\nTest MAE when splitting at different years:')
print(robust.round(3).to_string(index=False))


NAVY = '#1E2656'
PURPLE = '#514A75'
MAUVE = '#A68E99'
CORAL = '#F2836B'
PEACH = '#F7C79A'
INK = '#1E2656'

installed_fonts = [f.name for f in font_manager.fontManager.ttflist]
chart_font = 'Calibri' if 'Calibri' in installed_fonts else 'Carlito' if 'Carlito' in installed_fonts else 'DejaVu Sans'

plt.rcParams.update({'font.family': chart_font, 'font.size': 11,
                     'axes.titlesize': 14, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK, 'legend.frameon': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white', 'grid.color': '#E9E6EE', 'grid.alpha': 1,
                     'axes.axisbelow': True})

months = test['Month'].dt.to_timestamp()
bar_width = 0.26

strategy = pd.DataFrame({'Month': months})
for name, pred in [('Baseline', y_pred_base), ('Lasso', y_pred_lasso), ('Gradient Boosting', y_pred_gb)]:
    monthly = np.where(np.array(pred) > 0, y_test, 0) / 100
    strategy[name] = 10000 * np.cumprod(1 + monthly)
strategy.to_csv('stage2_model/outputs/investment_growth.csv', index=False)
print('\n----- $10,000 invested only in months the model predicts "up" (2019-2023) -----')
for name in ['Baseline', 'Lasso', 'Gradient Boosting']:
    print('  {:<18} ${:,.0f}'.format(name, strategy[name].iloc[-1]))


def plot_actual():
    plt.bar(months, y_test, width=22, color=PEACH)
    plt.axhline(0, color=INK, linewidth=0.8)
    plt.title('Actual Monthly Returns, 2019\u20132023')
    plt.ylabel('Return (%)')
    plt.ylim(-16, 20)
    plt.grid(axis='y', linewidth=1.2)


def plot_model_predictions():
    plt.plot(months, y_pred_base, color=MAUVE, linewidth=2, linestyle='--', label='Baseline (average month)')
    plt.plot(months, y_pred_lasso, color=NAVY, linewidth=2, marker='o', markersize=3.5, label='Lasso')
    plt.plot(months, y_pred_gb, color=CORAL, linewidth=2, marker='s', markersize=3.5, label='Gradient Boosting')
    plt.legend(loc='upper left', ncol=3, fontsize=10)
    plt.axhline(0, color=INK, linewidth=0.8)
    plt.title('Predicted Monthly Returns (zoomed)')
    plt.ylabel('Predicted return (%)')
    plt.ylim(-4, 4.5)
    plt.xlim(months.iloc[0] - pd.Timedelta(days=30), months.iloc[-1] + pd.Timedelta(days=30))
    plt.grid(axis='y', linewidth=1.2)


def plot_errors(metric):
    if metric == 'MAE':
        title = 'Prediction Error (MAE)'
    else:
        title = 'Prediction Error (RMSE)'
    periods = ['Training\n(2001\u20132018, cross-validated)', 'Testing\n(2019\u20132023)']
    xticks_index = range(0, 2)
    plt.xticks(xticks_index, periods)
    models = [('Baseline', MAUVE, 'Baseline (average month)'),
              ('Lasso', NAVY, 'Lasso'),
              ('Gradient Boosting', CORAL, 'Gradient Boosting')]
    for k, (label, colour, row) in enumerate(models):
        values = [train_cv_results.loc[row, 'Train_CV_' + metric], results.loc[row, metric]]
        positions = [x + (k - 1) * bar_width for x in xticks_index]
        plt.bar(positions, values, bar_width, color=colour, label=label)
        for x, v in zip(positions, values):
            plt.text(x, v + 0.08, '{:.2f}'.format(v), ha='center', fontsize=10)
    plt.ylim(0, 7)
    plt.title(title)
    plt.ylabel(metric + ' (percentage points)')
    plt.grid(axis='y', linewidth=1.2)
    plt.legend(loc='upper left', fontsize=10)


def plot_ffr_pattern():
    trends = ['Falling', 'Same', 'Rising']
    train_means = [by_direction.loc[('Train 2001-2018', t), 'mean'] for t in trends]
    test_means = [by_direction.loc[('Test 2019-2023', t), 'mean'] for t in trends]
    width = 0.36
    xticks_index = range(0, 3)
    plt.xticks(xticks_index, ['FFR fell\n(down > 0.10 pp)', 'FFR unchanged\n(within \u00b10.10 pp)', 'FFR rose\n(up > 0.10 pp)'])
    train_index = [x - width / 2 for x in xticks_index]
    test_index = [x + width / 2 for x in xticks_index]
    plt.bar(train_index, train_means, width, color=PEACH, label='Training years (2001\u20132018)')
    plt.bar(test_index, test_means, width, color=PURPLE, label='Testing years (2019\u20132023)')
    train_n = [by_direction.loc[('Train 2001-2018', t), 'count'] for t in trends]
    test_n = [by_direction.loc[('Test 2019-2023', t), 'count'] for t in trends]
    for i in xticks_index:
        plt.text(train_index[i], train_means[i] + (0.12 if train_means[i] > 0 else -0.55),
                 '{:+.1f}%\nn={}'.format(train_means[i], train_n[i]), ha='center', fontsize=10)
        plt.text(test_index[i], test_means[i] + (0.12 if test_means[i] > 0 else -0.55),
                 '{:+.1f}%\nn={}'.format(test_means[i], test_n[i]), ha='center', fontsize=10)
    plt.axhline(0, color=INK, linewidth=0.8)
    plt.ylim(-2.8, 4.2)
    plt.title('Returns After FFR Moves')
    plt.xlabel('Change in the FFR over the previous 3 months (pp = percentage points)')
    plt.ylabel('Average next-month return (%)')
    plt.grid(axis='y', linewidth=1.2)
    plt.legend(loc='upper right', fontsize=10)


def plot_growth():
    for name, colour, style in [('Baseline', MAUVE, '--'), ('Lasso', NAVY, '-'), ('Gradient Boosting', CORAL, '-')]:
        label = 'Always invested (baseline)' if name == 'Baseline' else name
        plt.plot(months, strategy[name], color=colour, linewidth=2.4, linestyle=style, label=label)
        final = strategy[name].iloc[-1]
        plt.text(months.iloc[-1], final, '  ${:,.0f}'.format(final), color=colour,
                 va='center', fontsize=11, fontweight='bold')
    plt.axhline(10000, color=INK, linewidth=0.8)
    plt.title('Growth of $10,000, 2019\u20132023')
    plt.ylabel('Portfolio value ($)')
    plt.grid(axis='y', linewidth=1.2)
    plt.xlim(months.iloc[0], months.iloc[-1] + pd.Timedelta(days=330))
    plt.legend(loc='upper left', fontsize=10)


plt.figure(figsize=(11, 7))
plt.subplot(2, 1, 1)
plot_actual()
plt.xlim(months.iloc[0] - pd.Timedelta(days=30), months.iloc[-1] + pd.Timedelta(days=30))
plt.subplot(2, 1, 2)
plot_model_predictions()
plt.tight_layout()
plt.savefig('stage2_model/outputs/fig1_actual_vs_predicted.png', dpi=300)

plt.figure(figsize=(11, 4.8))
plt.subplot(1, 2, 1)
plot_errors('MAE')
plt.subplot(1, 2, 2)
plot_errors('RMSE')
plt.suptitle('Training vs Testing', fontsize=15, fontweight='bold', color=INK)
plt.tight_layout()
plt.savefig('stage2_model/outputs/fig2_error_comparison.png', dpi=300)

plt.figure(figsize=(7, 4.8))
plot_ffr_pattern()
plt.tight_layout()
plt.savefig('stage2_model/outputs/fig3_ffr_pattern.png', dpi=300)

plt.figure(figsize=(9, 4.8))
plot_growth()
plt.tight_layout()
plt.savefig('stage2_model/outputs/fig4_investment_growth.png', dpi=300)

plt.figure(figsize=(16, 11))
plt.subplot2grid((4, 2), (0, 0))
plot_actual()
plt.xlim(months.iloc[0] - pd.Timedelta(days=30), months.iloc[-1] + pd.Timedelta(days=30))
plt.subplot2grid((4, 2), (1, 0))
plot_model_predictions()
plt.subplot2grid((4, 2), (0, 1), rowspan=2)
plot_errors('MAE')
plt.subplot2grid((4, 2), (2, 0), rowspan=2)
plot_ffr_pattern()
plt.subplot2grid((4, 2), (2, 1), rowspan=2)
plot_growth()
plt.suptitle('Can the FFR Predict the S&P 500?', fontsize=18, fontweight='bold', color=INK)
plt.tight_layout()
plt.savefig('stage2_model/outputs/model_figures.png', dpi=300)

plt.close('all')
print('\nSaved all results and graphs in stage2_model/outputs - open the PNG files there to view them')
