import gradio as gr
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

def object_to_numeric(Dataframe,dict_list = None):
    columns = Dataframe.columns
    count_list_no,count_unique = 0,0
    if dict_list ==None:
        dict_list = []
        for column in (columns):
            dict_list.append(dict())
            if Dataframe[column].dtype=='object':
                # print(column)
                for x in Dataframe[column].unique():
                    dict_list[count_list_no][x] = count_unique
                    count_unique +=1
                count_unique = 0
            count_list_no +=1
    count = 0
    for column in (columns):
        if Dataframe[column].dtype=='object':
            Dataframe[column] = Dataframe[column].replace(to_replace=dict_list[count])
            Dataframe[column] = Dataframe[column].astype(float)
        count +=1
    return (Dataframe,dict_list)

def Check_null(Dataframe):
    columns = Dataframe.columns
    columns_containing_null = Dataframe.isnull().sum()
    for index,column in enumerate(columns_containing_null):
        if column>0:
            print(f"{columns[index]}---> {column} null values")

def Normalize(Dataframe):
    columns = Dataframe.columns
    for column in columns:
        if column != 'SalePrice':
            Dataframe[column] =  Dataframe[column]/np.max(Dataframe[column])
    return Dataframe


Train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
Train_data.drop('Id',axis=1,inplace=True)
Train_data,list_dict = object_to_numeric(Train_data)
Check_null(Train_data)
Train_data = Train_data.fillna(0)
Train_data = Normalize(Train_data)
Check_null(Train_data)

X_values = np.array(Train_data.drop('SalePrice',axis=1))
y_values = np.array(Train_data['SalePrice'])
X_train,X_test,y_train,y_test = train_test_split(X_values,y_values,test_size=0.1,random_state=42)
# print(X_train.shape)

knn_regressor = KNeighborsRegressor(4,metric='manhattan')
knn_regressor.fit(X_train, y_train)

columns = Train_data.columns
Dataframe = pd.DataFrame(Train_data.iloc[0:1,0:-1].astype('object'))
values = []
def Predict(*arg):
    global list_dict,Dataframe,values
    # print(Dataframe)
    arg = list(arg)
    # print(arg)
    for index,x in enumerate(arg):
        if x =='':
            arg[index] = np.nan
    # print(arg)
    values = arg
    columns = Dataframe.columns
    Dataframe.iloc[0] = arg
    # print(Dataframe)
    for x in columns:
        try:
            Dataframe[x]=pd.to_numeric(Dataframe[x])
        except:
            pass
    # print(Dataframe.dtypes)
    Dataframe,list_dict = object_to_numeric(Dataframe,list_dict)
    Check_null(Dataframe)
    Dataframe = Dataframe.fillna(0)
    # print("Filled Na")
    # Dataframe = Normalize(Dataframe)
    Check_null(Dataframe)
    # print(Dataframe)
    # print(Dataframe.shape)
    X_check = np.array(Dataframe)
    # print(X_check.shape)
    # X_check = np.expand_dims(X_check, axis=0)
    y_pred = knn_regressor.predict(X_check)
    return float(y_pred)
    return 'Passed'

inp = []
with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Column():
        columns = Train_data.columns
        for x in columns[0:-1]:
            inp.append(gr.Textbox(placeholder=f"Content of Column {x}",label=x))
        out = gr.Textbox(label='Output')
    btn = gr.Button("Run")
    btn.click(fn=Predict, inputs=inp, outputs=out)

demo.launch()
