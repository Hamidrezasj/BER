import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time





#Calling preprocessed data frame
file_path = r'epc-model.csv'
df = pd.read_csv(file_path)

X = df.drop(['ENERGY_CONSUMPTION_CURRENT', 'CURRENT_ENERGY_RATING','CO2'], axis=1)
Y_energy=df['ENERGY_CONSUMPTION_CURRENT']
Y_epc=df['CURRENT_ENERGY_RATING']
Y_co2=df['CO2']

#Home menu
menu=st.sidebar.radio("Menu",["Home","predictions","Retrofit"])
if menu=="Home":
     st.title('AI for Building energy retrofit')
     st.markdown("""
    <p style="font-size: 1.2em; color: #555555;">This software utilizes artificial intelligence to predict buildings' annual energy consumption, CO2 emission, and EPC (Energy Performance Certificate) 
                 label based on their features. Additionally, it can estimate the cost of various retrofits and analyze their impact on energy performance and EPC ratings. 
                 The model has been developed using the Energy Performance Certificate dataset for residential buildings in the UK, published by the Department for Levelling Up, 
                 Housing and Communities. </p> """, unsafe_allow_html=True)
     st.image("hr_image.jpeg", width=600)


if menu=="predictions":
    st.title('Building features')
    #select Boxes-categorical features
    categories_property_types = X['PROPERTY_TYPE'].unique()
    selected_property_type = st.selectbox("Select Property Type:", categories_property_types)

    categories_built_form = X['BUILT_FORM'].unique()
    selected_built_form = st.selectbox("Select built form:" , categories_built_form, help= 'mid-terrace has external walls on two opposite sides; enclosed mid terrace has an external wall on one side only; end-terrace has three external walls; enclosed end-terrace has two adjacent external walls.' )

    selected_floor_area=st.number_input("Please enter floor area of the property? (between 20 and 500 sqm)", min_value=20, max_value=500)

    selected_floor_height=st.number_input("Please enter the floor height? ", min_value=2.00, max_value=10.00)

    col1,col2= st.columns(2)
    categories_age_band=X['CONSTRUCTION_AGE_BAND'].unique()
    checkbox_age_band= col2.checkbox("New build")
    if checkbox_age_band:
         selected_age_band=col1.selectbox("Select construction year of the property: ", ['2012 onwards'])
    else:
         selected_age_band=col1.selectbox("Select construction year of the property: ", categories_age_band)
          

    categories_glazed_type = X['GLAZED_TYPE'].unique()
    selected_glazed_type= st.selectbox("select glazing type: ", categories_glazed_type)

    selected_multi_glazed_proportion=st.number_input("multi glazed area of the total glazed area of the property? (%) ", min_value=0, max_value=100)

    selected_glazed_area='Normal'

    col1,col2=st.columns(2)
    checkbox_value_glazing=col2.checkbox('Glazing area based on system default (construction age band and property type)')

    if checkbox_value_glazing:
         if selected_property_type in ['Flat','Maisonette'] and selected_age_band in ['before 1900','1900-1929','1930-1949'] and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.0801+5.580
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1950-1966' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.0341+8.562
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1967-1975' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.0717+6.560
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1976-1982' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.1199+1.975
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1983-1990' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.0510+4.554
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1991-1995' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.0813+3.744
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1996-2002' and selected_glazed_area=='Normal':
              glazing_area= selected_floor_area*0.1148+0.392
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and selected_glazed_area=='Normal':
              glazing_area= selected_floor_area*0.1148+0.392
         elif selected_property_type in ['House','Bungalow'] and selected_age_band in ['before 1900','1900-1929','1930-1949'] and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.1220+6.875
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1950-1966' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.1294+5.515
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1967-1975' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.1239+7.332
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1976-1982' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.1252+5.520
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1983-1990' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.1356+5.242
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1991-1995' and selected_glazed_area=='Normal':
              glazing_area=selected_floor_area*0.0948+6.534
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1996-2002' and selected_glazed_area=='Normal':
              glazing_area= selected_floor_area*0.1382-0.027
         elif selected_property_type in ['House','Bungalow'] and selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and selected_glazed_area=='Normal':
              glazing_area= selected_floor_area*0.1435+0.403
     
          
         col1.write(f"**calculated glazing area is {glazing_area:.2f} square meters**")
         
         
    else:
         glazing_area= col1.number_input('Please enter the the glazing area (sqm): ')
         if selected_property_type in ['Flat','Maisonette'] and selected_age_band in ['before 1900','1900-1929','1930-1949'] and glazing_area > 1.25*(selected_floor_area*0.0801+5.580):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1950-1966' and glazing_area > 1.25*(selected_floor_area*0.0341+8.562):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1967-1975' and glazing_area > 1.25*(selected_floor_area*0.0717+6.560):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1976-1982' and glazing_area > 1.25*(selected_floor_area*0.1199+1.975):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1983-1990' and glazing_area > 1.25*(selected_floor_area*0.0510+4.554):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1991-1995' and glazing_area > 1.25*(selected_floor_area*0.0813+3.744):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1996-2002' and glazing_area > 1.25*(selected_floor_area*0.1148+0.392):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and glazing_area > 1.25*(selected_floor_area*0.1148+0.392):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band in ['before 1900','1900-1929','1930-1949'] and glazing_area > 1.25*(selected_floor_area*0.1220+6.875):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1950-1966' and glazing_area > 1.25*(selected_floor_area*0.1294+5.515):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1967-1975' and glazing_area > 1.25*(selected_floor_area*0.1239+7.332):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1976-1982' and glazing_area > 1.25*(selected_floor_area*0.1252+5.520):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1983-1990' and glazing_area > 1.25*(selected_floor_area*0.1356+5.242):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1991-1995' and glazing_area > 1.25*(selected_floor_area*0.0948+6.534):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1996-2002' and glazing_area > 1.25*(selected_floor_area*0.1382-0.027):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and glazing_area > 1.25*(selected_floor_area*0.1435+0.403):
              selected_glazed_area='More Than Typical'
         elif selected_property_type in ['Flat','Maisonette'] and selected_age_band in ['before 1900','1900-1929','1930-1949'] and glazing_area < 0.75*(selected_floor_area*0.0801+5.580):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1950-1966' and glazing_area < 0.75*(selected_floor_area*0.0341+8.562):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1967-1975' and glazing_area < 0.75*(selected_floor_area*0.0717+6.560):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1976-1982' and glazing_area < 0.75*(selected_floor_area*0.1199+1.975):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1983-1990' and glazing_area < 0.75*(selected_floor_area*0.0510+4.554):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1991-1995' and glazing_area < 0.75*(selected_floor_area*0.0813+3.744):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band=='1996-2002' and glazing_area < 0.75*(selected_floor_area*0.1148+0.392):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['Flat', 'Maisonette'] and selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and glazing_area < 0.75*(selected_floor_area*0.1148+0.392):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band in ['before 1900','1900-1929','1930-1949'] and glazing_area < 0.75*(selected_floor_area*0.1220+6.875):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1950-1966' and glazing_area < 0.75*(selected_floor_area*0.1294+5.515):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1967-1975' and glazing_area < 0.75*(selected_floor_area*0.1239+7.332):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1976-1982' and glazing_area < 0.75*(selected_floor_area*0.1252+5.520):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1983-1990' and glazing_area < 0.75*(selected_floor_area*0.1356+5.242):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1991-1995' and glazing_area < 0.75*(selected_floor_area*0.0948+6.534):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band=='1996-2002' and glazing_area < 0.75*(selected_floor_area*0.1382-0.027):
              selected_glazed_area='Less Than Typical'
         elif selected_property_type in ['House','Bungalow'] and selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and glazing_area < 0.75*(selected_floor_area*0.1435+0.403):
              selected_glazed_area='Less Than Typical'

    
    categories_floor_type=X['FLOOR_TYPE'].unique()
    selected_floor_type=st.selectbox("select type of the floor: ", categories_floor_type)

    categories_floor_insulation=X['FLOOR_INSULATION'].unique()
    if selected_floor_type=='Another dwelling or premises below':
         selected_floor_insulation=st.selectbox("insulation in the floor: ",['Another dwelling or premises below'])
    elif selected_floor_type in ['Solid (next to the ground)','Suspended (next to the ground)','Exposed or to unheated space']:
         selected_floor_insulation=st.selectbox("insulation in the floor: ",['As built','Insulated-at least 50mm insulation'])
     
    col1,col2=st.columns(2)    
    selected_wall_u_value=col1.number_input('please enter U-Value of the external wall', min_value=0.00, max_value=10.00)
    checkbox_u_value= col2.checkbox('External wall U-value based on system default (construction age band, wall, and insulation type)')
    if checkbox_u_value:
         wall_type=col1.selectbox('Please enter external wall type:', ['Timber frame','Solid brick', 'Cavity wall'])
         if checkbox_age_band:
              wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built'])    
         elif wall_type=='Cavity wall':
             wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation', 'more than 100mm insulation',
                                                                                        'filled cavity','filled cavity with 50-99mm insulation',
                                                                                        'filled cavity with more than 100mm insulation'])
         elif wall_type=='Solid brick':
             wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation','100-149mm insulation', 'more than 150mm insulation'])
         elif wall_type=='Timber frame':
              wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','Timber frame with internal insulation'])


              

             
         if wall_type=='Solid brick' and wall_insulation=='As built' and selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              selected_wall_u_value=2.1
         elif wall_type=='Solid brick' and wall_insulation=='As built' and selected_age_band in ['1967-1975']:
              selected_wall_u_value=1.7
         elif wall_type=='Solid brick' and wall_insulation=='As built' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=1
         elif wall_type=='Solid brick' and wall_insulation=='As built' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.6
         elif wall_type=='Solid brick' and wall_insulation=='As built' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.45
         elif wall_type=='Solid brick' and wall_insulation=='As built' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.35
         elif wall_type=='Solid brick' and wall_insulation=='As built' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.30
         elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation' and selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              selected_wall_u_value=0.6
         elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1967-1975']:
              selected_wall_u_value=0.55
         elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.45
         elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.35
         elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.3
         elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.25
         elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.21
         elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation' and selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              selected_wall_u_value=0.35
         elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation' and selected_age_band in ['1967-1975']:
              selected_wall_u_value=0.35
         elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.32
         elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.24
         elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.21
         elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.19
         elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.17
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              selected_wall_u_value=0.25
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['1967-1975']:
              selected_wall_u_value=0.25
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.21
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.18
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.17
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.15
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.14
         elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation' and selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              selected_wall_u_value=0.25
         elif wall_type=='Cavity wall' and wall_insulation=='As built' and selected_age_band in ['before 1900']:
              selected_wall_u_value=2.1
         elif wall_type=='Cavity wall' and wall_insulation=='As built' and selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              selected_wall_u_value=1.6
         elif wall_type=='Cavity wall' and wall_insulation=='As built' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=1
         elif wall_type=='Cavity wall' and wall_insulation=='As built' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.6
         elif wall_type=='Cavity wall' and wall_insulation=='As built' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.45
         elif wall_type=='Cavity wall' and wall_insulation=='As built' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.35
         elif wall_type=='Cavity wall' and wall_insulation=='As built' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.30
         elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation' and selected_age_band in ['before 1900']:
              selected_wall_u_value=0.6
         elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              selected_wall_u_value=0.53
         elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.45
         elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.35
         elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.3
         elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.25
         elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.21
         elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation' and selected_age_band in ['before 1900']:
              selected_wall_u_value=0.35
         elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation' and selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              selected_wall_u_value=0.32
         elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.3
         elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.24
         elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.21
         elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.19
         elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.17
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity' and selected_age_band in ['before 1900']:
              selected_wall_u_value=0.5
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity' and selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              selected_wall_u_value=0.5
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.4
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.35
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.35
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.35
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.3
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation' and selected_age_band in ['before 1900']:
              selected_wall_u_value=0.31
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation' and selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              selected_wall_u_value=0.31
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.27
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.25
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.25
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.25
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.21
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation' and selected_age_band in ['before 1900']:
              selected_wall_u_value=0.22
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation' and selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              selected_wall_u_value=0.22
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.20
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation' and selected_age_band in ['1983-1990','1991-1995']:
              selected_wall_u_value=0.19
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation' and selected_age_band in ['1996-2002']:
              selected_wall_u_value=0.19
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.19
         elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.16
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['before 1900']:
              selected_wall_u_value=2.5
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['1900-1929','1930-1949']:
              selected_wall_u_value=1.9
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['1950-1966']:
              selected_wall_u_value=1
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['1967-1975']:
              selected_wall_u_value=0.8
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.45
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['1983-1990','1991-1995','1996-2002']:
              selected_wall_u_value=0.4
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.35
         elif wall_type=='Timber frame' and wall_insulation=='As built' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.3
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['before 1900']:
              selected_wall_u_value=0.6
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['1900-1929','1930-1949']:
              selected_wall_u_value=0.55
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['1950-1966']:
              selected_wall_u_value=0.4
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['1967-1975']:
              selected_wall_u_value=0.4
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['1976-1982']:
              selected_wall_u_value=0.4
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['1983-1990','1991-1995','1996-2002']:
              selected_wall_u_value=0.4
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['2003-2006']:
              selected_wall_u_value=0.35
         elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation' and selected_age_band in ['2007-2011','2012 onwards']:
              selected_wall_u_value=0.3
         
          
         col1.write(f"**calculated external wall U-value is {selected_wall_u_value:.2f} W/sqm**")

    else: 
         wall_type=col1.selectbox('Please enter external wall type:', ['Timber frame','Solid brick', 'Cavity wall'])
         if checkbox_age_band:
              wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built'])
         elif wall_type=='Cavity wall':
             wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation', 'more than 100mm insulation',
                                                                                        'filled cavity','filled cavity with 50-99mm insulation',
                                                                                        'filled cavity with more than 100mm insulation'])
         elif wall_type=='Solid brick':
             wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation','100-149mm insulation', 'more than 150mm insulation'])
         elif wall_type=='Timber frame':
              wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','Timber frame with internal insulation'])
     
     
         
         
     
    col1,col2=st.columns(2)
    checkbox_value_wall=col2.checkbox('external wall area based on system default (property type, built form, floor area, floor height, and glazing area)')
    flat_width=(selected_floor_area/2)**0.5
    flat_length=2*flat_width

    if checkbox_value_wall:
         if selected_property_type in ['Flat','Maisonette'] and selected_built_form=='Mid-Terrace':
              wall_length=(flat_length*selected_floor_height)
              external_wall_area=(2*wall_length)-glazing_area
         elif selected_property_type in ['Flat','Maisonette'] and selected_built_form=='Enclosed Mid-Terrace':
              wall_length=(flat_length*selected_floor_height)
              external_wall_area=(wall_length)-glazing_area
         elif selected_property_type in ['Flat','Maisonette'] and selected_built_form=='End-Terrace':
              wall_width=flat_width*selected_floor_height
              wall_length=flat_length*selected_floor_height
              external_wall_area=(2*wall_width)+wall_length-glazing_area
         elif selected_property_type in ['Flat','Maisonette'] and selected_built_form=='Enclosed End-Terrace':
              wall_width=flat_width*selected_floor_height
              wall_length=flat_length*selected_floor_height
              external_wall_area=wall_width+wall_length-glazing_area
         elif selected_property_type in ['House', 'Bungalow'] and selected_built_form=='Detached':
              wall_width=flat_width*selected_floor_height
              external_wall_area=(8*wall_width)-glazing_area
         elif selected_property_type in ['House', 'Bungalow'] and selected_built_form=='Semi-Detached':
              wall_width=flat_width*selected_floor_height
              external_wall_area=(6*wall_width)-glazing_area
          
         col1.write(f"**calculated wall area is {external_wall_area:.2f} square meters**")

    else:
         external_wall_area=col1.number_input('please enter the area of the external wall')
    
    categories_roof_type=X['ROOF_TYPE'].unique()
    selected_roof_type= st.selectbox("select type of roof: ", categories_roof_type)

    if selected_roof_type=='Another dwelling or premises above':
         selected_roof_insulation=st.selectbox('Select type of roof insulation: ',['Another dwelling or premises above'])
    elif selected_roof_type=='Pitched Roof':
         selected_roof_insulation=st.selectbox('Select type of roof insulation: ',['As built','less than 50mm loft insulation', '50 to 99mm loft insulation',
                                                                                 '100 to 200mm loft insulation','More than 200mm loft insulation'])
    elif selected_roof_type in ['Flat roof', 'Roof room']:
         selected_roof_insulation=st.selectbox('Select type of roof insulation: ',['As built','Insulated-unknown thickness (50mm or more)'])

    
    

    categories_heating_system=X['HEATING_SYSTEM'].unique()
    if checkbox_age_band:
         selected_heating_system=st.selectbox("Select type of main heating system: ", ['Air source heat pump with radiators or underfloor heating','Air source heat pump with warm air distribution' ])
    else:
         selected_heating_system=st.selectbox("Select type of main heating system: ", categories_heating_system)

    categories_hotwater=X['HOTWATER_DESCRIPTION'].unique()
    selected_hotwater= st.selectbox("select your hotwater system: ", categories_hotwater )

    categories_secondary_heating=X['SECONDHEAT_DESCRIPTION'].unique()
    selected_secondary_heating=st.selectbox("Select your secondary heating system: " , categories_secondary_heating )

    categories_main_fuel=X['MAIN_FUEL'].unique()
    selected_main_fuel= st.selectbox("what is the main fuel of energy system? ", categories_main_fuel)

    categories_ventilation=X['MECHANICAL_VENTILATION'].unique()
    selected_ventilation=st.selectbox("select type of ventilation system", categories_ventilation)

    selected_low_energy_lighting=st.number_input("percentage of low energy lighting in the property? (%) ", min_value=0, max_value=100)

    categories_solar_hotwater=X['SOLAR_WATER_HEATING_FLAG'].unique()
    selected_solar_hotwater=st.selectbox('whether the hotwater in the Property is from solar', categories_solar_hotwater)
    
    max_pv=float(round(0.12*(selected_floor_area/2)*(1/0.819),1))
    st.session_state['max_pv']=max_pv
    installed_capacity_pv=st.number_input('Please enter installed photovoltaic capacity (KW)', min_value=0.00 , max_value=max_pv)
    pv_area=installed_capacity_pv*8.333
    roof_area=1
    if selected_roof_type=='Pitched Roof' and selected_property_type in ['House','Bungalow']:
         roof_area=(selected_floor_area/2)/0.819
    elif selected_roof_type=='Flat roof' and selected_property_type in ['House','Bungalow']: 
         roof_area=(selected_floor_area/2)
    else:
         pv_area=0
     
    selected_pv_supply=(pv_area/roof_area)*100

    st.session_state['roof_area']=roof_area
    st.session_state['installed_pv']=installed_capacity_pv

    
    

    #my case study dataframe
    my_case_study = pd.DataFrame({
     'PROPERTY_TYPE':[selected_property_type],  #cat
     'BUILT_FORM': [selected_built_form],   #cat
     'TOTAL_FLOOR_AREA':[selected_floor_area],   #num
     'MULTI_GLAZE_PROPORTION': [selected_multi_glazed_proportion],   #num
     'GLAZED_TYPE': [selected_glazed_type],   #cat
     'GLAZED_AREA': [selected_glazed_area],   #cat
     'LOW_ENERGY_LIGHTING': [selected_low_energy_lighting],   #num
     'HOTWATER_DESCRIPTION': [selected_hotwater],   #cat
     'SECONDHEAT_DESCRIPTION': [selected_secondary_heating],   #cat
     'MAIN_FUEL': [selected_main_fuel],   #cat
     'FLOOR_HEIGHT': [selected_floor_height],   #num
     'PHOTO_SUPPLY': [selected_pv_supply],   #num
     'SOLAR_WATER_HEATING_FLAG': [selected_solar_hotwater],   #cat
     'MECHANICAL_VENTILATION': [selected_ventilation],   #cat
     'CONSTRUCTION_AGE_BAND': [selected_age_band],   #cat
     'HEATING_SYSTEM': [selected_heating_system],   #cat
     'FLOOR_TYPE': [selected_floor_type],   #cat
     'FLOOR_INSULATION': [selected_floor_insulation],   #cat
     'WALLS_U_VALUE': [selected_wall_u_value],   #num
     'ROOF_TYPE': [selected_roof_type],   #cat
     'ROOF_INSULATION': [selected_roof_insulation],   #cat
    
    })


    st.session_state['building_features']=my_case_study
    st.session_state['glazing']=glazing_area
    st.session_state['wall']=external_wall_area
    st.session_state['wall_type']=wall_type
    st.session_state['wall_insulation']=wall_insulation


    categorical_columns= ['PROPERTY_TYPE','BUILT_FORM','GLAZED_TYPE', 'GLAZED_AREA','HOTWATER_DESCRIPTION','SECONDHEAT_DESCRIPTION',
                          'MAIN_FUEL','MECHANICAL_VENTILATION','CONSTRUCTION_AGE_BAND','HEATING_SYSTEM','SOLAR_WATER_HEATING_FLAG',
                          'FLOOR_TYPE','FLOOR_INSULATION','ROOF_TYPE','ROOF_INSULATION']  
    numerical_columns= ['TOTAL_FLOOR_AREA', 'MULTI_GLAZE_PROPORTION','LOW_ENERGY_LIGHTING','FLOOR_HEIGHT','PHOTO_SUPPLY','WALLS_U_VALUE'] 
    
    st.write('')
    st.write('')
    st.write('')

    #scaling
    scaler=MinMaxScaler()
    X_scaled=X.copy()
    my_case_study_scaled=my_case_study.copy()
    X_scaled[numerical_columns]=scaler.fit_transform(X[numerical_columns])
    my_case_study_scaled[numerical_columns]=scaler.transform(my_case_study[numerical_columns])

    #encoding
    X_encoded_scaled=pd.get_dummies(X_scaled)
    my_case_study_encoded_scaled=pd.get_dummies(my_case_study_scaled).reindex(columns=X_encoded_scaled.columns, fill_value=0)
    encoder=LabelEncoder()
    Y_epc_encoded=encoder.fit_transform(Y_epc)


    X_encoded_scaled_train, X_encoded_scaled_test, Y_energy_train, Y_energy_test= train_test_split(X_encoded_scaled,Y_energy,test_size=0.2, random_state=100)
    X_encoded_scaled_train, X_encoded_scaled_test, Y_epc_encoded_train, Y_epc_encoded_test= train_test_split(X_encoded_scaled,Y_epc_encoded,test_size=0.2,random_state=100)
    X_encoded_scaled_train, X_encoded_scaled_test, Y_co2_train, Y_co2_test= train_test_split(X_encoded_scaled,Y_co2,test_size=0.2,random_state=100)
    
    #creating XGBoost model
    @st.cache_resource(ttl=0.5*3600)
    def XGBmodel(X_encoded_scaled_train,Y_energy_train):
        model=XGBRegressor()
        model.fit(X_encoded_scaled_train,Y_energy_train)
        return model
    
    @st.cache_resource(ttl=0.5*3600)
    def XGBmodel_epc(X_encoded_scaled_train,Y_epc_encoded_train):
         model_epc=XGBClassifier()
         model_epc.fit(X_encoded_scaled_train,Y_epc_encoded_train)
         return model_epc
    
    @st.cache_resource(ttl=0.5*3600)
    def XGBmodel_co2(X_encoded_scaled_train,Y_co2_train):
         model_co2=XGBRegressor()
         model_co2.fit(X_encoded_scaled_train,Y_co2_train)
         return model_co2
      
    #cross val scores
    #model_temp=XGBClassifier()
    #cv_score1=cross_val_score(model_temp,X_encoded_scaled,Y_epc_encoded,cv=10,scoring='accuracy')
    #st.write(cv_score1)
    #model_temp2=XGBRegressor()
    #cv_score2=cross_val_score(model_temp2,X_encoded_scaled,Y_energy,cv=10,scoring='r2')
    #cv_score3=cross_val_score(model_temp2,X_encoded_scaled,Y_co2,cv=10,scoring='r2')
    #st.write(cv_score2)
    #st.write(cv_score3)

    model=XGBmodel(X_encoded_scaled_train,Y_energy_train)
    model_epc=XGBmodel_epc(X_encoded_scaled_train,Y_epc_encoded_train)
    model_co2=XGBmodel_co2(X_encoded_scaled_train,Y_co2_train)

    st.session_state['ml_model']=model
    st.session_state['ml_model_epc']=model_epc
    st.session_state['ml_model_co2']=model_co2
    st.session_state['encoding']=my_case_study_encoded_scaled


    #predictions
    y_predicted_energy=model.predict(my_case_study_encoded_scaled)
    y_pred_XGB=model.predict(X_encoded_scaled_test)
    y_pred_epc=model_epc.predict(my_case_study_encoded_scaled)
    y_pred_co2=model_co2.predict(my_case_study_encoded_scaled)
    y_co2_accuracy=model_co2.predict(X_encoded_scaled_test)
    y_epc_accuracy=model_epc.predict(X_encoded_scaled_test)
    #r2 and accuracy scores
    r2_co2=r2_score(Y_co2_test,y_co2_accuracy)
    r2_energy=r2_score(Y_energy_test,y_pred_XGB)
    accuracy_epc=accuracy_score(Y_epc_encoded_test,y_epc_accuracy)
    #st.write(r2_co2)
    #st.write(r2_energy)
    #st.write(accuracy_epc)

    epc_mapping={
          0:'B',
          1:'C',
          2:'D',
          3:'E',
          4:'F',
          5:'G'
    }


    st.session_state['predicted_energy_consumption']=y_predicted_energy
    st.session_state['predicted_co2']=y_pred_co2
    st.session_state['predicted_epc']=epc_mapping[int(y_pred_epc)]




    #image_path="D:\PhD-UWL\codes\main\epc1.jpeg"

    #triggering prediction
    col1, col2, col3=st.columns(3)

    if col2.button("Predict annual CO2 emission"):
            col2.write(f"**CO2 emissions per square metre floor area per year is {int(y_pred_co2)}** (Kg/sqm)")
     
    if col3.button("Predict annual energy consumption"):
            col3.write(f"**Annual primary energy consumption is estimated {int(y_predicted_energy)}** (KWh/sqm)")
     
    
    if col1.button("predict EPC rating"):
         if y_pred_epc==0:
              col1.image("epc-b.jpg",width=400)
         elif y_pred_epc==1:
              col1.image("epc-c.jpg",width=400)
         elif y_pred_epc==2:
              col1.image("epc-d.jpg",width=400)
         elif y_pred_epc==3: 
              col1.image("epc-e.jpg",width=400)
         elif y_pred_epc==4:
              col1.image("epc-f.jpg",width=400)
         elif y_pred_epc==5:
              col1.image("epc-g.jpg",width=400)

    
     

    
     
####################################################################################################################################################################

#Retrofit options 



if menu=='Retrofit':
     st.title('Retrofit options')
     st.write('')
     st.write('')
     st.write('')
     my_case_study=st.session_state['building_features']
     external_wall_area=st.session_state['wall']
     glazing_area=st.session_state['glazing']
     wall_type= st.session_state['wall_type']
     wall_insulation=st.session_state['wall_insulation']
     model_energy_retrofit=st.session_state['ml_model']
     model_epc_retrofit=st.session_state['ml_model_epc']
     model_co2_retrofit=st.session_state['ml_model_co2']
     my_case_study_encoded_scaled=st.session_state['encoding']
     max_pv=st.session_state['max_pv']
     roof_area=st.session_state['roof_area']
     installed_capacity_pv=st.session_state['installed_pv']
     y_predicted_energy=st.session_state['predicted_energy_consumption']
     y_predicted_epc=st.session_state['predicted_epc']
     y_predicted_co2=st.session_state['predicted_co2']

     my_case_study_retrofitted=my_case_study.copy()

     col1,col2=st.columns(2)
     col1.write('__1-Externall wall retrofit__')

     #Retrofit prices    #Retrofit prices    #Retrofit prices               #Retrofit prices            #Retrofit prices             #Retrofit prices           #Retrofit prices
     wall_insulation_50mm=90            #GBP
     wall_insulation_100mm=110          #GBP
     wall_insulation_150mm=130          #GBP
     wall_insulation_200mm=150          #GBP
     cavity_filling=20                  #GBP
     double_glazed_window= 540          #GBP
     triple_glazed_window= 1200         #GBP
     suspended_floor_insulation=105     #GBP
     solid_floor_insulation= 80         #GBP
     roof_insulation_25mm=17.5          #GBP
     roof_insulation_50mm=20            #GBP
     roof_insulation_100mm=25           #GBP
     roof_insulation_200mm=35           #GBP

     #ASHP quote      #ASHP quote           #ASHP quote           #ASHP quote           #ASHP quote          #ASHP quote          #ASHP quote            #ASHP quote          #ASHP quote

     @st.cache_resource(ttl=0.5*3600)
     def ashp_quote(quote_features):
          PATH = "chromedriver.exe"

          chrome_options= webdriver.ChromeOptions()
          chrome_options.add_argument("--headless")


          driver=webdriver.Chrome(options=chrome_options)
          driver.get('http://asf-hp-cost-demo-l-b-1046547218.eu-west-1.elb.amazonaws.com/')

          drop_down_1= Select(driver.find_element(By.NAME,"region"),)
          drop_down_1.select_by_visible_text(quote_features['region'])

          drop_down_2= Select(driver.find_element(By.NAME,"premisesInfo.type"),)
          drop_down_2.select_by_visible_text(quote_features['premises_type'])

          drop_down_3= Select(driver.find_element(By.NAME,"premisesInfo.builtForm"),)
          drop_down_3.select_by_visible_text(quote_features['built_form'])

          drop_down_3= Select(driver.find_element(By.NAME,"premisesInfo.ageBand"),)
          drop_down_3.select_by_visible_text(quote_features['age_band'])

          input_1=driver.find_element(By.NAME,"premisesInfo.numRooms")
          input_1.send_keys(quote_features['num_rooms'])

          input_2=driver.find_element(By.NAME,"premisesInfo.floorArea")
          input_2.send_keys(quote_features['floor_area'])

          submit=driver.find_element(By.XPATH,"//button[normalize-space()='Submit']")
          submit.click()

          waiter=WebDriverWait(driver,10)
          waiter.until(EC.presence_of_element_located((By.CSS_SELECTOR,"body > div:nth-child(2) > div:nth-child(3) > div:nth-child(3) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > span:nth-child(5) > span:nth-child(6)")))

          cost_text=driver.find_element(By.CSS_SELECTOR,"body > div:nth-child(2) > div:nth-child(3) > div:nth-child(3) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > span:nth-child(5) > span:nth-child(6)")
          cost_str=str(cost_text.text)
          cost_str=cost_str.replace(',','')
          cost_str=cost_str.replace('£','')
          cost_int=int(cost_str)
          cost_int_updated=1.20*cost_int

          return cost_int_updated
     

     floor_area=str(my_case_study['TOTAL_FLOOR_AREA'].iloc[0])
     num_rooms=round((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/40)+1
     num_rooms=str(num_rooms)
     quote_features={
          'region':'London',
          'premises_type': 'House',
          'built_form': 'Detached',
          'age_band': '1930-1965',
          'num_rooms': num_rooms,
          'floor_area': floor_area
     }

     ashp_price_quote=ashp_quote(quote_features)

     
     df_wall_u_value=pd.read_csv(r'wall-u-value.csv')
     df_wall_u_value.set_index('Unnamed: 0',inplace=True)


     if wall_type=='Solid brick' and wall_insulation=='As built':
     
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall', '100mm insulation for external wall', '150mm insulation for external wall',
                                                                          'No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(wall_insulation_50mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 50mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='100mm insulation for external wall':
               retrofit_cost=int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='150mm insulation for external wall':
               retrofit_cost=int(wall_insulation_150mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 150mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Solid brick' and wall_insulation=='50-99mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('100mm insulation for external wall', '150mm insulation for external wall','No retrofit is required'))
          if wall_option=='100mm insulation for external wall':
               retrofit_cost=int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='150mm insulation for external wall':
               retrofit_cost=int(wall_insulation_150mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 150mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Solid brick' and wall_insulation=='100-149mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('150mm insulation for external wall','No retrofit is required'))
          if wall_option=='150mm insulation for external wall':
               retrofit_cost=int(wall_insulation_150mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 150mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Solid brick' and wall_insulation=='more than 150mm insulation':
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Timber frame' and wall_insulation=='As built':
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall','No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(wall_insulation_50mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['timber frame- internal insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
     elif wall_type=='Timber frame' and wall_insulation=='Timber frame with internal insulation':
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Cavity wall' and wall_insulation=='As built':
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall', '100mm insulation for external wall', 'Filling cavity'
                                                                          ,'Filling cavity and 50mm insulation','Filling cavity and 100mm insulation','No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(wall_insulation_50mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['unfilled cavity- 50mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='100mm insulation for external wall':
               retrofit_cost=int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['unfilled cavity- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity':
               retrofit_cost=int(cavity_filling*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity and 50mm insulation':
               retrofit_cost=int(cavity_filling*external_wall_area)+int(wall_insulation_50mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 50mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity and 100mm insulation':
               retrofit_cost=int(cavity_filling*external_wall_area)+int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Cavity wall' and wall_insulation=='50-99mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('100mm insulation for external wall', 'Filling cavity','Filling cavity and 100mm insulation','No retrofit is required'))
          if wall_option=='100mm insulation for external wall':
               retrofit_cost=int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['unfilled cavity- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity':
               retrofit_cost=int(cavity_filling*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 50mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity and 100mm insulation':
               retrofit_cost=int(cavity_filling*external_wall_area)+int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Cavity wall' and wall_insulation=='more than 100mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('Filling cavity','No retrofit is required'))
          if wall_option=='Filling cavity':
               retrofit_cost=int(cavity_filling*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
     
     elif wall_type=='Cavity wall' and wall_insulation=='filled cavity':
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall', '100mm insulation for external wall','No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(wall_insulation_50mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 50mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='100mm insulation for external wall':
               retrofit_cost=int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with 50-99mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('100mm insulation for external wall','No retrofit is required'))
          if wall_option=='100mm insulation for external wall':
               retrofit_cost=int(wall_insulation_100mm*external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif wall_type=='Cavity wall' and wall_insulation=='filled cavity with more than 100mm insulation':
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
     

     col1,col2=st.columns(2)
     col1.write('__2-Retrofit glazing__')
     if my_case_study['GLAZED_TYPE'].iloc[0] in ['Secondary glazing','Single glazing']:
          glazing_option=col1.radio('Choose glazing type: ',('Double glazing','Triple glazing', 'No retrofit is required'))
          if glazing_option=='Double glazing':
               retrofit_cost_2=int(glazing_area*double_glazed_window)
               col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
               my_case_study_retrofitted['GLAZED_TYPE']='Double glazing'
          elif glazing_option=='Triple glazing':
               retrofit_cost_2=int(glazing_area*triple_glazed_window)
               col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
               my_case_study_retrofitted['GLAZED_TYPE']='Triple glazing'
          elif glazing_option=='No retrofit is required':
               retrofit_cost_2=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
     elif my_case_study['GLAZED_TYPE'].iloc[0] in ['Double glazing','Triple glazing']:
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_2=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
     
     col1,col2=st.columns(2)
     col1.write('__3-Floor retrofit__')
     if (my_case_study['FLOOR_TYPE'].iloc[0]=='Another dwelling or premises below') or (my_case_study['FLOOR_TYPE'].iloc[0] in 
                                                                                      ['Solid (next to the ground)','Suspended (next to the ground)','Exposed or to unheated space'] and 
                                                                                      my_case_study['FLOOR_INSULATION'].iloc[0]=='Insulated-at least 50mm insulation'):
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_3=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
     elif (my_case_study['FLOOR_TYPE'].iloc[0]=='Suspended (next to the ground)' and my_case_study['FLOOR_INSULATION'].iloc[0]=='As built'):
          floor_option= col1.radio('Choose floor insulation: ',['50mm insulation for floor','No retrofit is required'])
          if floor_option=='50mm insulation for floor':
              retrofit_cost_3=(my_case_study['TOTAL_FLOOR_AREA'].iloc[0])*suspended_floor_insulation
              col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
              my_case_study_retrofitted['FLOOR_INSULATION']='Insulated-at least 50mm insulation'
          elif floor_option=='No retrofit is required':
               retrofit_cost_3=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
     elif (my_case_study['FLOOR_TYPE'].iloc[0] in ['Solid (next to the ground)','Exposed or to unheated space'] and my_case_study['FLOOR_INSULATION'].iloc[0]=='As built'):
          floor_option= col1.radio('Choose floor insulation: ',['50mm insulation for floor','No retrofit is required'])
          if floor_option=='50mm insulation for floor':
              retrofit_cost_3=(my_case_study['TOTAL_FLOOR_AREA'].iloc[0])*solid_floor_insulation
              col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
              my_case_study_retrofitted['FLOOR_INSULATION']='Insulated-at least 50mm insulation'
          elif floor_option=='No retrofit is required':
               retrofit_cost_3=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')


     col1,col2=st.columns(2)
     col1.write('__4-Roof retrofit__')
     if (my_case_study['ROOF_TYPE'].iloc[0]=='Another dwelling or premises above') or (my_case_study['ROOF_TYPE'].iloc[0] in ['Flat roof', 'Roof room'] and 
                                                                                       my_case_study['ROOF_INSULATION'].iloc[0]=='Insulated-unknown thickness (50mm or more)') or (my_case_study['ROOF_TYPE'].iloc[0] =='Pitched Roof' and my_case_study['ROOF_INSULATION'].iloc[0]=='More than 200mm loft insulation'):
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_4=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (my_case_study['ROOF_INSULATION'].iloc[0]=='As built'):
          roof_option=col1.radio('Choose roof insulation: ', ['25mm loft insulation','50mm loft insulation','100mm loft insulation','200mm loft insulation','No retrofit is required'])
          if roof_option=='25mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_25mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='less than 50mm loft insulation'
          elif roof_option=='50mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='50 to 99mm loft insulation'
          elif roof_option=='100mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_100mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='100 to 200mm loft insulation'
          elif roof_option=='200mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (my_case_study['ROOF_INSULATION'].iloc[0]=='less than 50mm loft insulation'):
          roof_option=col1.radio('Choose roof insulation: ', ['50mm loft insulation','100mm loft insulation','200mm loft insulation','No retrofit is required'])
          if roof_option=='50mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='50 to 99mm loft insulation'
          elif roof_option=='100mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_100mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='100 to 200mm loft insulation'
          elif roof_option=='200mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (my_case_study['ROOF_INSULATION'].iloc[0]=='50 to 99mm loft insulation'):
          roof_option=col1.radio('Choose roof insulation: ', ['100mm loft insulation','200mm loft insulation','No retrofit is required'])
          if roof_option=='100mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_100mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='100 to 200mm loft insulation'
          elif roof_option=='200mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (my_case_study['ROOF_INSULATION'].iloc[0]=='100 to 200mm loft insulation'):
          roof_option=col1.radio('Choose roof insulation: ', ['200mm loft insulation','No retrofit is required'])
          if roof_option=='200mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (my_case_study['ROOF_TYPE'].iloc[0]=='Flat roof') and (my_case_study['ROOF_INSULATION'].iloc[0]=='As built'):
          roof_option=col1.radio('Choose roof insulation: ', ['50mm insulation','No retrofit is required'])
          if roof_option=='50mm loft insulation':
               retrofit_cost_4=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='Insulated-unknown thickness (50mm or more)'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (my_case_study['ROOF_TYPE'].iloc[0]=='Roof room') and (my_case_study['ROOF_INSULATION'].iloc[0]=='As built'):
          roof_option=col1.radio('Choose roof insulation: ', ['50mm loft insulation','No retrofit is required'])
          if roof_option=='50mm loft insulation':
               roof_room_floor=((my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)
               roof_room_wall=((roof_room_floor*0.3*0.66)**0.5)*8.25
               retrofit_cost_4=(roof_room_floor+roof_room_wall)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4:.2f} GBP**')
               my_case_study_retrofitted['ROOF_INSULATION']='Insulated-unknown thickness (50mm or more)'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')


     col1,col2=st.columns(2)
     col1.write('__5-Heating system retrofit__')
     if my_case_study['HEATING_SYSTEM'].iloc[0] in ['Boiler system with radiators or underfloor heating','Electric storage system','Electric underfloor heating',
                                                    'Room heater','Warm air system (not heat pump)']:
          heating_option=col1.radio('Choose heating system retrofit: ',['Air source heat pump','No retrofit is required'])
          if heating_option=='Air source heat pump':
               col2.write(f'Air source heat pump cost (assuming no need of radiator upgrades): {int(ashp_price_quote)} GBP')
               gov_grant=7500
               col2.write(f'Boiler upgrade scheme government grant: {gov_grant} GBP')
               retrofit_cost_5=ashp_price_quote-gov_grant
               col2.write(f'**Estimated total cost after government grant is {int(retrofit_cost_5)} GBP**')
               my_case_study_retrofitted['HEATING_SYSTEM'] = 'Air source heat pump with radiators or underfloor heating'
               my_case_study_retrofitted['MAIN_FUEL']='All electric'
               
          elif heating_option=='No retrofit is required':
               retrofit_cost_5=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_5} GBP**')

     elif my_case_study ['HEATING_SYSTEM'].iloc[0] in ['Air source heat pump with radiators or underfloor heating','Air source heat pump with warm air distribution']:
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_5=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_5} GBP**')
     

     st.write('')
     col1,col2=st.columns(2)
     col1.write('__6-Adding PV solar to the property__')
     pv_retrofit=col1.number_input('Please enter required PV capacity (KWp) :', min_value=0.00 , max_value=max_pv-installed_capacity_pv)
     pv_area_retrofit=pv_retrofit*8.333
     my_case_study_retrofitted['PHOTO_SUPPLY']=((pv_area_retrofit/roof_area)*100) + my_case_study['PHOTO_SUPPLY'].iloc[0]
     if pv_retrofit<=4:
          retrofit_cost_6=pv_retrofit*2393
     elif 4 < pv_retrofit <= 10:
          retrofit_cost_6=pv_retrofit*2216
     elif 10 < pv_retrofit <=50:
          retrofit_cost_6=pv_retrofit*1502

     
     col2.write(f'**estimated retrofit cost is {int(retrofit_cost_6)} GBP**')
     




     

     st.write('')
     st.write('')
     st.write('')
     st.write('')
     total_retrofit_cost=retrofit_cost+retrofit_cost_2+retrofit_cost_3+retrofit_cost_4+retrofit_cost_5+retrofit_cost_6
     st.write(f'**Total retrofit cost is {int(total_retrofit_cost)} GBP**')

     st.write('')
     st.write('')

     categorical_columns= ['PROPERTY_TYPE','BUILT_FORM','GLAZED_TYPE', 'GLAZED_AREA','HOTWATER_DESCRIPTION','SECONDHEAT_DESCRIPTION',
                          'MAIN_FUEL','MECHANICAL_VENTILATION','CONSTRUCTION_AGE_BAND','HEATING_SYSTEM','SOLAR_WATER_HEATING_FLAG',
                          'FLOOR_TYPE','FLOOR_INSULATION','ROOF_TYPE','ROOF_INSULATION']  
     numerical_columns= ['TOTAL_FLOOR_AREA', 'MULTI_GLAZE_PROPORTION','LOW_ENERGY_LIGHTING','FLOOR_HEIGHT','PHOTO_SUPPLY','WALLS_U_VALUE']

     scaler=MinMaxScaler()
     X_scaled=X.copy()
     X_scaled[numerical_columns]=scaler.fit_transform(X[numerical_columns])
     my_case_study_retrofitted_scaled=my_case_study_retrofitted.copy()
     my_case_study_retrofitted_scaled[numerical_columns]=scaler.transform(my_case_study_retrofitted[numerical_columns])

     my_case_study_retrofitted_encoded_scaled=pd.get_dummies(my_case_study_retrofitted_scaled).reindex( columns=my_case_study_encoded_scaled.columns , fill_value=0)



     y_pred_epc_retrofitted=model_epc_retrofit.predict(my_case_study_retrofitted_encoded_scaled)
     y_pred_energy_retrofitted=model_energy_retrofit.predict(my_case_study_retrofitted_encoded_scaled)
     y_pred_co2_retrofitted=model_co2_retrofit.predict(my_case_study_retrofitted_encoded_scaled)

     epc_mapping={
          0:'B',
          1:'C',
          2:'D',
          3:'E',
          4:'F',
          5:'G'
    }

     col1, col2, col3= st.columns(3)
     if col3.button("predict annual energy consumption"):
            col3.write(f"annual energy consumption before retrofit was **{int(y_predicted_energy)}** and after retrofit is estimated **{int(y_pred_energy_retrofitted)}** (KWh/sqm)")
     
     if col2.button("predict annual CO2 emission"):
            col2.write(f"CO2 emissions per square metre floor area per year before retrofit was **{int(y_predicted_co2)}** and after retrofit is estimated **{int(y_pred_co2_retrofitted)}** (Kg/sqm)")
     

     if col1.button('predict EPC rating'):
          col1.write(f'EPC rating before retrofit was **{y_predicted_epc}** and after retrofit is **{epc_mapping[int(y_pred_epc_retrofitted)]}**')

     






     



     
          






     







    



