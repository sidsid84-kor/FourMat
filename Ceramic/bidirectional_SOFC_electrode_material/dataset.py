import ipywidgets as widgets
import codecs
import pandas as pd
import io
from IPython.display import clear_output

class UI_load_data:
    def __init__(self):
        self.file_selector = widgets.FileUpload(button_style='primary', description = 'csv파일')
        self.return_button = widgets.Button()
        display(self.file_selector)


    def get_data(self):
        try:
            uploaded_file = self.file_selector.value[0]
            uploaded_file.content.tobytes()
            codecs.decode(uploaded_file.content, encoding="utf-8")
            self.df = pd.read_csv(io.BytesIO(uploaded_file.content))
        except:
            idx=list(self.file_selector.value.keys())[0]
            uploaded_file = self.file_selector.value[idx]
            codecs.decode(uploaded_file['content'], encoding="utf-8")
            self.df = pd.read_csv(io.BytesIO(uploaded_file['content']))
        return self.df

    def head(self, n=5):
        print(self.df.head(n=n))

