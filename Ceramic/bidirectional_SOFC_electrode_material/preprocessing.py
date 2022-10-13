import ipywidgets as widgets
from IPython.display import clear_output

class UI_set_data:
    def __init__(self, df):
        self.df = df
        feature = list(df.corr().columns)
        self.X_checklist = [widgets.Checkbox(value=False, description=label) for label in feature]
        self.y_checklist = widgets.Dropdown(options=feature)
        X_checkboxes = widgets.GridBox(children=self.X_checklist, layout=widgets.Layout(width='100%', grid_template_columns='33% 33% 33%'))
        return_button = widgets.Button(description='Return', button_style='primary')
        print('=============== X 변수(입력데이터) 선택 ===============')
        display(X_checkboxes)
        print('=============== y 변수(타겟데이터) 선택 ===============')
        display(self.y_checklist)
        display(return_button)
        return_button.on_click(self.__return_value_on_click__)


    def __return_value_on_click__(self, b):
        clear_output(wait=True)
        self.X_column = [x.description for x in self.X_checklist if x.value]
        if self.y_checklist.value in self.X_column:
            print('동일한 X변수와 y변수를 선택할 수 없습니다.')
            self.X_column = []
            self.y_column = []
        else:
            self.y_column = [self.y_checklist.value]
            print(f'선택된 X변수 : {self.X_column}')
            print(f'선택된 y변수 : {self.y_column}')
            print('self.X, self.y로 값이 할당되었습니다.')

class Split_data:
    def __init__(self, df, X , y):
        self.df = df
        self.X = X
        self.y = y
        self.train_ratio = widgets.IntSlider(
            value=80,
            min=0,
            max=100,
            step=5,
            description='학습 데이터 비율(%) : ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=widgets.Layout(width='50%'),
            style={'description_width': 'initial'}
        )
        return_button = widgets.Button(description='Return', button_style='primary')
        display(self.train_ratio)
        display(return_button)
        return_button.on_click(self.__return_value_on_click__)

    def __return_value_on_click__(self, b):
        train_ratio = int(self.train_ratio.value)
        train_num = len(self.df) * (train_ratio / 100)
        self.train_X = self.df.loc[:train_num, self.X]
        self.train_y = self.df.loc[:train_num, self.y]
        self.test_X = self.df.loc[train_num::, self.X]
        self.test_y = self.df.loc[train_num:, self.y]
        clear_output(wait=True)
        print('train_X, train_y, test_X, test_y = self.get_splited_data()를 사용하여 할당받으세요.')
    def get_splited_data(self):
        return self.train_X, self.train_y, self.test_X, self.test_y