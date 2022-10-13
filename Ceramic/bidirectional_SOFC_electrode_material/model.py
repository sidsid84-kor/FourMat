import pandas as pd
import ipywidgets as widgets
from pycaret.regression import *
import plotly.graph_objs as go

class UI_regressor_model:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.model_dict = {'Linear Regression': 'lr', 'Lasso Regression': 'lasso', 'Ridge Regression': 'ridge', 'Elastic Net': 'en', 'Least Angle Regression': 'lar', 'Lasso Least Angle Regression': 'llar', 'Orthogonal Matching Pursuit': 'omp',
                           'Bayesian Ridge': 'br', 'Automatic Relevance Determination': 'ard', 'Passive Aggressive Regressor': 'par', 'Random Sample Consensus': 'ransac', 'TheilSen Regressor': 'tr', 'Huber Regressor': 'huber',
                           'Kernel Ridge': 'kr', 'Support Vector Regression': 'svm', 'K Neighbors Regressor': 'knn', 'Decision Tree Regressor': 'dt', 'Random Forest Regressor': 'rf', 'Extra Trees Regressor': 'et', 'AdaBoost Regressor': 'ada',
                           'Gradient Boosting Regressor': 'gbr', 'MLP Regressor': 'mlp', 'Extreme Gradient Boosting': 'xgboost', 'Light Gradient Boosting Machine': 'lightgbm', 'Dummy Regressor': 'dummy'}

        self.y_column = test_y.columns.tolist()[0]
        X = pd.concat([train_X, test_X], axis=0)
        y = pd.concat([train_y, test_y], axis=0)
        df = pd.concat([X, y], axis = 1)
        next1_button = widgets.Button(description='Next', button_style='primary')
        train_ratio = len(train_y) / (len(train_y) + len(test_y))
        self.set_up = setup(data = df, target = self.y_column, fold = 5, silent = True,
                            train_size = train_ratio, verbose=False)


        self.model_selector = widgets.Dropdown(options=self.model_dict.keys())
        next1_button.on_click(self.__next1_on_click__)
        display(self.model_selector)
        display(next1_button)

    #         ignore_features = [i for i in self.df.columns if i not in self.y], verbose=True
    def __next1_on_click__(self, b):
        result = compare_models(sort = 'R2')
        remove_metric('RMSLE')
        model_compare_result = pull().round(2)
        print(self.model_selector.value)
        model = model_compare_result['Model'][0]
        trained_model = create_model(self.model_dict[self.model_selector.value], verbose = False)
        result = predict_model(trained_model, verbose=False)
        value_max, value_min = max(list(result[self.y_column])), min(list(result[self.y_column]))
        fig_size = [value_min-(value_max-value_min)/10, value_max+(value_max-value_min)/10]
        fig = go.Figure()
        fig.update_layout(title = self.model_selector.value, title_font_size = 28, title_x = 0.5, plot_bgcolor = '#FFFFFF', width = 600, height = 600, xaxis_title = '실험값', yaxis_title = '모델 예측값', xaxis_range = fig_size, yaxis_range = fig_size)
        fig.update_xaxes(showline=True, mirror=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, mirror=True, linewidth=2, linecolor='black')
        fig.add_trace(go.Scatter(x = list(result.loc[:, self.y_column]), y = list(result.loc[:, 'Label']), mode='markers', name = str(self.y_column), marker=dict(color="#0AA344", opacity=0.5, size = 8)))
        fig.add_trace(go.Scatter(x = fig_size, y = fig_size, name = 'base', mode = 'lines',line = dict(color = 'black', dash = 'dot')))

        fig.add_annotation(text = 'R2 score : {:.3f}'.format(model_compare_result.loc[self.model_dict[model], 'R2']), x = fig_size[1], y = fig_size[0], xref="paper", yref="paper", showarrow = False, align = 'left', font = dict(size = 15))
        fig.add_annotation(text = 'RMSE : {:.3f}'.format(model_compare_result.loc[self.model_dict[model], 'RMSE']), x = fig_size[1], y = fig_size[0], xref="paper", yref="paper", showarrow = False, align = 'left', font = dict(size = 15))
        fig.show()