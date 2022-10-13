import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output
from sklearn import preprocessing
import plotly.express as px

def UI_show_corr(df):
    cor_col = df.corr().columns
    temp_data = df[cor_col]
    feature = [x for x in df.columns if x in list(cor_col)]
    checkboxes = [widgets.Checkbox(value=True, description=label) for label in feature]



    selected_feature = [x.description for x in checkboxes if x.value]
    button = widgets.Button(description='Next', button_style='primary')
    widgets_ = widgets.GridBox(children=checkboxes, layout=widgets.Layout(width = '100%', grid_template_columns='33% 33% 33%'))
    display(widgets_)
    display(button)
    def click_event(b):
        clear_output(wait=True)
        selected_feature = [x.description for x in widgets_.children if x.value]
        normalizaion = preprocessing.StandardScaler().fit(temp_data)
        nor_final = pd.DataFrame(normalizaion.transform(temp_data))
        nor_final.columns = cor_col
        corr = nor_final.corr().round(2)
        corr.fillna('0', inplace=True)
        fig = px.imshow(corr.loc[selected_feature, selected_feature], text_auto = True, color_continuous_scale=px.colors.sequential.RdBu, zmin = -1, zmax = 1)
        fig.update_layout(title = '상관관계 분석', title_font_size = 30, title_x = 0.5, width = 750, height = 750, plot_bgcolor = '#FFFFFF')
        fig.show()
        reset_button = widgets.Button(description='Reset', button_style='success')
        display(reset_button)
        def reset_event(b):
            clear_output(wait=True)
            return UI_show_corr(df=df, feature=feature)
        reset_button.on_click(reset_event)
    button.on_click(click_event)
