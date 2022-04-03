from flask import Flask, render_template, request
from model import InputForm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load model and scaler
model = keras.models.load_model('C:/Users/Kuzmins/Documents/Jupiter/VKRfinal/models/VKR_mlp_var1')
scaler = joblib.load('C:/Users/Kuzmins/Documents/Jupiter/VKRfinal/scaler.pkl')
#Created DF
dfuser = pd.DataFrame({'Sootnoshenie_Matrix_napolnitel' : [0.0], 'Plotnost_kg/m3' : [0.0],
            'Modul_Uprugosti_GPa' : [0.0], 'Kolichestvo_napolnitela_m%' : [0.0],
            'Soderjanie_epoxidnih_grupp_%_2' : [0.0], 'Temp_vspishki_C_2' : [0.0],
            'Poverhnostnaia_plotnost_g/m2' : [0.0], 'Modul_uprugosti_pri_rastiajenii_GPa' : [0.0],
            'Prochnost_pri_rastiajenii_Mpa' : [0.0], 'Potreblenie_smoli_g/m2' : [0.0],
            'Ugol_nashivki_grad' : [0.0], 'Shag_nashivki' : [0.0], 'Plotnost_nashivki' : [0.0]})

col = dfuser.columns

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        Field1 = form.Field1.data
        Field2 = form.Field2.data
        Field3 = form.Field3.data
        Field4 = form.Field4.data
        Field5 = form.Field5.data
        Field6 = form.Field6.data
        Field7 = form.Field7.data
        Field8 = form.Field8.data
        Field9 = form.Field9.data
        Field10 = form.Field10.data



        # Processing data and entry to DF

        dfuser['Plotnost_kg/m3'].values[0] = float(Field1)
        dfuser['Modul_Uprugosti_GPa'].values[0] = float(Field2)
        dfuser['Kolichestvo_napolnitela_m%'].values[0] = float(Field3)
        dfuser['Soderjanie_epoxidnih_grupp_%_2'].values[0] = float(Field4)
        dfuser['Temp_vspishki_C_2'].values[0] = float(Field5)
        dfuser['Poverhnostnaia_plotnost_g/m2'].values[0] = float(Field6)
        dfuser['Potreblenie_smoli_g/m2'].values[0] = float(Field7)
        dfuser['Ugol_nashivki_grad'].values[0] = float(Field8)
        dfuser['Shag_nashivki'].values[0] = float(Field9)
        dfuser['Plotnost_nashivki'].values[0] = float(Field10)

        # Нормализация with MinMaxScaler

        modified = scaler.transform(dfuser)
        dfuser_modified = pd.DataFrame(modified, columns=col)
        dfuser_modified.drop(
            ['Sootnoshenie_Matrix_napolnitel', 'Modul_uprugosti_pri_rastiajenii_GPa', 'Prochnost_pri_rastiajenii_Mpa'],
            inplace=True, axis=1)
        dfuser_modified

        #Prediction

        pred = model.predict(dfuser_modified)

        pred_inversed = pd.DataFrame([])
        pred_inversed.at[0, 0] = float(pred[0][0])

        for i in range(1, 13):
            p = 0
            pred_inversed.at[0, i] = float(p)
        pred_inversed.at[0, 7] = float(pred[0][1])
        pred_inversed.at[0, 8] = float(pred[0][2])

        # Возвращает предсказанное значание (в размерности до нормализации)
        Y_trans = scaler.inverse_transform(pred_inversed)

        print(Y_trans[0, 0], Y_trans[0, 7], Y_trans[0, 8])

        s1 = Y_trans[0, 0]
        s2 = Y_trans[0, 7]
        s3 = Y_trans[0, 8]

    else:
        s1 = None
        s2 = None
        s3 = None

    return render_template("/view.html", form=form, s1=s1, s2=s2, s3=s3)


if __name__ == '__main__':
    app.run(debug=True)