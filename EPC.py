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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score,f1_score
from sklearn.model_selection import cross_val_score,StratifiedKFold,KFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
#import seaborn as sns
from imblearn.pipeline import make_pipeline

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver import FirefoxOptions
import time
import os,sys
#from streamlit_extras.stylable_container import stylable_container





#Calling preprocessed data frame
file_path = r'epc-model.csv'
df = pd.read_csv(file_path)

X = df.drop(['ENERGY_CONSUMPTION_CURRENT', 'CURRENT_ENERGY_RATING','CO2','MAIN_FUEL'], axis=1)
Y_energy=df['ENERGY_CONSUMPTION_CURRENT']
Y_epc=df['CURRENT_ENERGY_RATING']
Y_co2=df['CO2']


# Set default menu
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

# Sidebar controlled by session_state
menu = st.sidebar.radio("Main Menu", 
    ["Home", "Layout", "Dimensions", "Glazing system", 
     "Building envelope", "HVAC system", "Lighting", 
     "Renewable generation","Energy Performance","Retrofit"],
    index=["Home", "Layout", "Dimensions", "Glazing system", 
           "Building envelope", "HVAC system", "Lighting", 
           "Renewable generation","Energy Performance","Retrofit"].index(st.session_state.menu)
)

menu_options=["Home", "Layout", "Dimensions", "Glazing system", "Building envelope", "HVAC system", "Lighting", "Renewable generation","Energy Performance","Retrofit"]


# Update session_state if user selects something manually
if menu != st.session_state.menu:
    st.session_state.menu = menu
    st.rerun()

# Page content
if st.session_state.menu == "Home":
    st.title('AI for Building energy retrofit')
    st.markdown("""
    <p style="font-size: 1.2em; color: #555555;">This software utilizes artificial intelligence to predict buildings' annual energy consumption, CO2 emission, and EPC (Energy Performance Certificate) 
                 label based on their features. Additionally, it can estimate the cost of various retrofits and analyze their impact on energy performance and EPC ratings. 
                 The model has been developed using the Energy Performance Certificate dataset for residential buildings in the UK, published by the Department for Levelling Up, 
                 Housing and Communities. </p> """, unsafe_allow_html=True)
    st.image("hr_image.jpeg", width=400)

    if st.button('Start Energy Performance Assessment'):
        st.session_state.menu = menu_options[1]
        st.rerun()

elif st.session_state.menu== menu_options[1]:
    st.title("1-Building Layout")
    categories_property_types = X['PROPERTY_TYPE'].unique()
    st.session_state.selected_property_type = st.selectbox("Select Property Type:", categories_property_types, index=list(X['PROPERTY_TYPE'].unique()).index("Flat"))

    categories_built_form = X['BUILT_FORM'].unique()
    st.session_state.selected_built_form = st.selectbox("Select built form:" , categories_built_form, 
                                       help= 'mid-terrace has external walls on two opposite sides; enclosed mid terrace has an external wall on one side only; end-terrace has three external walls; enclosed end-terrace has two adjacent external walls.',
                                         index=list(X['BUILT_FORM'].unique()).index("Mid-Terrace"))
    
    col1, col2, col3=st.columns(3)
    if col3.button('Next'):
        st.session_state.menu = menu_options[2]
        st.rerun()
    
    elif col1.button('Previous'):
        st.session_state.menu = menu_options[0]
        st.rerun()
        
elif st.session_state.menu== menu_options[2]:
    st.title('2-Building Dimensions')
    st.session_state.selected_floor_area=st.number_input("Please enter floor area of the property? (between 20 and 500 sqm)", min_value=20, max_value=500, value=80)

    st.session_state.selected_floor_height=st.number_input("Please enter the floor to ceiling height? ", min_value=2.00, max_value=10.00, value=2.6)

    col1,col2= st.columns(2)
    categories_age_band=X['CONSTRUCTION_AGE_BAND'].unique()
    st.session_state.checkbox_age_band= col2.checkbox("New build")
    if st.session_state.checkbox_age_band:
        st.session_state.selected_age_band=col1.selectbox("Select construction year of the property: ", ['2012 onwards'])
    else:
        st.session_state.selected_age_band=col1.selectbox("Select construction year of the property: ", categories_age_band,index=list(X['CONSTRUCTION_AGE_BAND'].unique()).index("1983-1990"))
    
    col1, col2, col3=st.columns(3)
    if col3.button('Next'):
        st.session_state.menu = menu_options[3]
        st.rerun()
    
    elif col1.button('Previous'):
        st.session_state.menu = menu_options[1]
        st.rerun()


elif st.session_state.menu== menu_options[3]:
    st.title('3-Building Glazing System')

    categories_glazed_type = X['GLAZED_TYPE'].unique()
    st.session_state.selected_glazed_type= st.selectbox("select glazing type: ", categories_glazed_type)

    st.session_state.selected_multi_glazed_proportion=st.number_input("What percentage of the total window area in the property is multi-glazed? (%) ", min_value=0, max_value=100, value=100)

    st.session_state.selected_glazed_area='Normal'

    col1,col2=st.columns(2)
    checkbox_value_glazing=col2.checkbox('Glazing area based on system default (construction age band and property type)',value=True)

    if checkbox_value_glazing:
         if st.session_state.selected_property_type in ['Flat','Maisonette'] and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949'] and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.0801+5.580
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1950-1966' and st.session_state.elected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.0341+8.562
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1967-1975' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.0717+6.560
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1976-1982' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.1199+1.975
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1983-1990' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.0510+4.554
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1991-1995' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.0813+3.744
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1996-2002' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area= st.session_state.selected_floor_area*0.1148+0.392
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area= st.session_state.selected_floor_area*0.1148+0.392
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949'] and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.1220+6.875
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1950-1966' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.1294+5.515
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1967-1975' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.1239+7.332
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1976-1982' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.1252+5.520
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1983-1990' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.1356+5.242
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1991-1995' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area=st.session_state.selected_floor_area*0.0948+6.534
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1996-2002' and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area= st.session_state.selected_floor_area*0.1382-0.027
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and st.session_state.selected_glazed_area=='Normal':
              st.session_state.glazing_area= st.session_state.selected_floor_area*0.1435+0.403
     
          
         col1.write(f"**Calculated glazing area is {st.session_state.glazing_area:.2f} square meters**")

         
         
    else:
         st.session_state.glazing_area= col1.number_input('Please enter the the glazing area (sqm): ')
         if st.session_state.selected_property_type in ['Flat','Maisonette'] and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949'] and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.0801+5.580):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1950-1966' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.0341+8.562):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1967-1975' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.0717+6.560):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1976-1982' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1199+1.975):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1983-1990' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.0510+4.554):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1991-1995' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.0813+3.744):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1996-2002' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1148+0.392):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1148+0.392):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949'] and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1220+6.875):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1950-1966' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1294+5.515):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1967-1975' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1239+7.332):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1976-1982' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1252+5.520):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1983-1990' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1356+5.242):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1991-1995' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.0948+6.534):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1996-2002' and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1382-0.027):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and st.session_state.glazing_area > 1.25*(st.session_state.selected_floor_area*0.1435+0.403):
              st.session_state.selected_glazed_area='More Than Typical'
         elif st.session_state.selected_property_type in ['Flat','Maisonette'] and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949'] and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.0801+5.580):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1950-1966' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.0341+8.562):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1967-1975' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.0717+6.560):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1976-1982' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1199+1.975):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1983-1990' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.0510+4.554):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1991-1995' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.0813+3.744):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band=='1996-2002' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1148+0.392):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['Flat', 'Maisonette'] and st.session_state.selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1148+0.392):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949'] and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1220+6.875):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1950-1966' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1294+5.515):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1967-1975' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1239+7.332):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1976-1982' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1252+5.520):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1983-1990' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1356+5.242):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1991-1995' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.0948+6.534):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band=='1996-2002' and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1382-0.027):
              st.session_state.selected_glazed_area='Less Than Typical'
         elif st.session_state.selected_property_type in ['House','Bungalow'] and st.session_state.selected_age_band in ['2003-2006','2007-2011','2012 onwards'] and st.session_state.glazing_area < 0.75*(st.session_state.selected_floor_area*0.1435+0.403):
              st.session_state.selected_glazed_area='Less Than Typical'


    col1, col2, col3=st.columns(3)
    if col3.button('Next'):
        st.session_state.menu = menu_options[4]
        st.rerun()
    
    elif col1.button('Previous'):
        st.session_state.menu = menu_options[2]
        st.rerun()

    
elif  st.session_state.menu== menu_options[4]:
    st.title('4-Building Envelope')

    categories_floor_type=X['FLOOR_TYPE'].unique()
    st.session_state.selected_floor_type=st.selectbox("Select type of the floor: ", categories_floor_type, index=list(X['FLOOR_TYPE'].unique()).index("Suspended (next to the ground)"))

    categories_floor_insulation=X['FLOOR_INSULATION'].unique()
    if st.session_state.selected_floor_type=='Another dwelling or premises below':
         st.session_state.selected_floor_insulation=st.selectbox("insulation in the floor: ",['Another dwelling or premises below'])
    elif st.session_state.selected_floor_type in ['Solid (next to the ground)','Suspended (next to the ground)','Exposed or to unheated space']:
         st.session_state.selected_floor_insulation=st.selectbox("insulation in the floor: ",['As built','Insulated-at least 50mm insulation'])
     
    col1,col2=st.columns(2)    
    st.session_state.selected_wall_u_value=col1.number_input('Please enter U-Value of the external wall', min_value=0.00, max_value=10.00)
    checkbox_u_value= col2.checkbox('External wall U-value based on system default (construction age band, wall, and insulation type)', value=True)
    if checkbox_u_value:
         st.session_state.wall_type=col1.selectbox('Please enter external wall type:', ['Timber frame','Solid brick', 'Cavity wall'],index=2)
         if st.session_state.checkbox_age_band:
              st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built'])    
         elif st.session_state.wall_type=='Cavity wall':
             st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation', 'more than 100mm insulation',
                                                                                        'filled cavity','filled cavity with 50-99mm insulation',
                                                                                        'filled cavity with more than 100mm insulation'])
         elif st.session_state.wall_type=='Solid brick':
             st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation','100-149mm insulation', 'more than 150mm insulation'])
         elif st.session_state.wall_type=='Timber frame':
              st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','Timber frame with internal insulation'])


              

             
         if st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              st.session_state.selected_wall_u_value=2.1
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1967-1975']:
              st.session_state.selected_wall_u_value=1.7
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=1
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.6
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.45
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.30
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              st.session_state.selected_wall_u_value=0.6
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1967-1975']:
              st.session_state.selected_wall_u_value=0.55
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.45
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.3
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.21
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation' and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation' and st.session_state.selected_age_band in ['1967-1975']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.32
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.24
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.21
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.19
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.17
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['1967-1975']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.21
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.18
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.17
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.15
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.14
         elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation' and st.session_state.selected_age_band in ['before 1900','1900-1929','1930-1949','1950-1966']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=2.1
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              st.session_state.selected_wall_u_value=1.6
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=1
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.6
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.45
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.30
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=0.6
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              st.session_state.selected_wall_u_value=0.53
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.45
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.3
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.21
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation' and st.session_state.selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              st.session_state.selected_wall_u_value=0.32
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.3
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.24
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.21
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.19
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.17
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=0.5
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity' and st.session_state.selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              st.session_state.selected_wall_u_value=0.5
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.4
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.3
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=0.31
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation' and st.session_state.selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              st.session_state.selected_wall_u_value=0.31
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.27
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.25
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.21
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=0.22
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation' and st.session_state.selected_age_band in ['1900-1929','1930-1949','1950-1966','1967-1975']:
              st.session_state.selected_wall_u_value=0.22
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.20
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995']:
              st.session_state.selected_wall_u_value=0.19
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation' and st.session_state.selected_age_band in ['1996-2002']:
              st.session_state.selected_wall_u_value=0.19
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.19
         elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.16
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=2.5
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1900-1929','1930-1949']:
              st.session_state.selected_wall_u_value=1.9
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1950-1966']:
              st.session_state.selected_wall_u_value=1
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1967-1975']:
              st.session_state.selected_wall_u_value=0.8
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.45
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['1983-1990','1991-1995','1996-2002']:
              st.session_state.selected_wall_u_value=0.4
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.3
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['before 1900']:
              st.session_state.selected_wall_u_value=0.6
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['1900-1929','1930-1949']:
              st.session_state.selected_wall_u_value=0.55
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['1950-1966']:
              st.session_state.selected_wall_u_value=0.4
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['1967-1975']:
              st.session_state.selected_wall_u_value=0.4
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['1976-1982']:
              st.session_state.selected_wall_u_value=0.4
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['1983-1990','1991-1995','1996-2002']:
              st.session_state.selected_wall_u_value=0.4
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['2003-2006']:
              st.session_state.selected_wall_u_value=0.35
         elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation' and st.session_state.selected_age_band in ['2007-2011','2012 onwards']:
              st.session_state.selected_wall_u_value=0.3
         
          
         col1.write(f"**Calculated external wall U-value is {st.session_state.selected_wall_u_value:.2f} W/sqm**")

    else: 
         st.session_state.wall_type=col1.selectbox('Please enter external wall type:', ['Timber frame','Solid brick', 'Cavity wall'])
         if st.session_state.checkbox_age_band:
              st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built'])
         elif st.session_state.wall_type=='Cavity wall':
             st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation', 'more than 100mm insulation',
                                                                                        'filled cavity','filled cavity with 50-99mm insulation',
                                                                                        'filled cavity with more than 100mm insulation'])
         elif st.session_state.wall_type=='Solid brick':
             st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','50-99mm insulation','100-149mm insulation', 'more than 150mm insulation'])
         elif st.session_state.wall_type=='Timber frame':
              st.session_state.wall_insulation=col2.selectbox('Please enter external wall insulation: ', ['As built','Timber frame with internal insulation'])
     
     
         
         
     
    col1,col2=st.columns(2)
    checkbox_value_wall=col2.checkbox('External wall area based on system default (property type, built form, floor area, floor height, and glazing area)',value=True)
    flat_width=((st.session_state.selected_floor_area/2)**0.5)+0.6    #external wall thickness is 0.3
    flat_length=(2*flat_width)+0.6

    if checkbox_value_wall:
         if st.session_state.selected_property_type in ['Flat','Maisonette'] and st.session_state.selected_built_form=='Mid-Terrace':
              wall_length=(flat_length*(st.session_state.selected_floor_height+0.1))
              st.session_state.external_wall_area=(2*wall_length)-st.session_state.glazing_area
         elif st.session_state.selected_property_type in ['Flat','Maisonette'] and st.session_state.selected_built_form=='Enclosed Mid-Terrace':
              wall_length=(flat_length*(st.session_state.selected_floor_height+0.1))
              st.session_state.external_wall_area=(wall_length)-st.session_state.glazing_area
         elif st.session_state.selected_property_type in ['Flat','Maisonette'] and st.session_state.selected_built_form=='End-Terrace':
              wall_width=flat_width*(st.session_state.selected_floor_height+0.1)
              wall_length=flat_length*(st.session_state.selected_floor_height+0.1)
              st.session_state.external_wall_area=(2*wall_width)+wall_length-st.session_state.glazing_area
         elif st.session_state.selected_property_type in ['Flat','Maisonette'] and st.session_state.selected_built_form=='Enclosed End-Terrace':
              wall_width=flat_width*(st.session_state.selected_floor_height+0.1)
              wall_length=flat_length*(st.session_state.selected_floor_height+0.1)
              st.session_state.external_wall_area=wall_width+wall_length-st.session_state.glazing_area
         elif st.session_state.selected_property_type in ['House', 'Bungalow'] and st.session_state.selected_built_form=='Detached':
              wall_width=flat_width*(st.session_state.selected_floor_height+0.3)
              st.session_state.external_wall_area=(8*wall_width)-st.session_state.glazing_area
         elif st.session_state.selected_property_type in ['House', 'Bungalow'] and st.session_state.selected_built_form=='Semi-Detached':
              wall_width=flat_width*(st.session_state.selected_floor_height+0.3)
              st.session_state.external_wall_area=(6*wall_width)-st.session_state.glazing_area
          
         col1.write(f"**Calculated wall area is {st.session_state.external_wall_area:.2f} square meters**")

    else:
         st.session_state.external_wall_area=col1.number_input('Please enter the area of the external wall')
    
    categories_roof_type=X['ROOF_TYPE'].unique()
    st.session_state.selected_roof_type= st.selectbox("Select type of roof: ", categories_roof_type, index=list(X['ROOF_TYPE'].unique()).index("Another dwelling or premises above"))

    if st.session_state.selected_roof_type=='Another dwelling or premises above':
         st.session_state.selected_roof_insulation=st.selectbox('Select type of roof insulation: ',['Another dwelling or premises above'])
    elif st.session_state.selected_roof_type=='Pitched Roof':
         st.session_state.selected_roof_insulation=st.selectbox('Select type of roof insulation: ',['As built','less than 50mm loft insulation', '50 to 99mm loft insulation',
                                                                                 '100 to 200mm loft insulation','More than 200mm loft insulation'])
    elif st.session_state.selected_roof_type in ['Flat roof', 'Roof room']:
         st.session_state.selected_roof_insulation=st.selectbox('Select type of roof insulation: ',['As built','Insulated-unknown thickness (50mm or more)'])

    
    col1, col2, col3=st.columns(3)
    if col3.button('Next'):
        st.session_state.menu = menu_options[5]
        st.rerun()
    
    elif col1.button('Previous'):
        st.session_state.menu = menu_options[3]
        st.rerun()


elif  st.session_state.menu== menu_options[5]:
    st.title('5-Building HVAC')


    categories_heating_system=X['HEATING_SYSTEM'].unique()
    if st.session_state.checkbox_age_band:
         st.session_state.selected_heating_system=st.selectbox("Select type of main heating system: ", ['Air source heat pump with radiators or underfloor heating','Air source heat pump with warm air distribution' ])
    else:
         st.session_state.selected_heating_system=st.selectbox("Select type of main heating system: ", categories_heating_system)

    categories_hotwater=X['HOTWATER_DESCRIPTION'].unique()
    st.session_state.selected_hotwater= st.selectbox("Select your hot water system: ", categories_hotwater )

    categories_secondary_heating=X['SECONDHEAT_DESCRIPTION'].unique()
    st.session_state.selected_secondary_heating=st.selectbox("Select your secondary heating system: " , categories_secondary_heating )

    categories_ventilation=X['MECHANICAL_VENTILATION'].unique()
    st.session_state.selected_ventilation=st.selectbox("Select type of ventilation system", categories_ventilation)


    col1, col2, col3=st.columns(3)
    if col3.button('Next'):
        st.session_state.menu = menu_options[6]
        st.rerun()
    
    elif col1.button('Previous'):
        st.session_state.menu = menu_options[4]
        st.rerun()


elif  st.session_state.menu== menu_options[6]:
    st.title('6-Lighting System')

    st.session_state.selected_low_energy_lighting=st.number_input("Percentage of low energy lighting in the property? (%) ",
                                                                   min_value=0, max_value=100, value=100, help='What percentage of lighting in the property is low-energy (such as LED)? ')
    
    col1, col2, col3=st.columns(3)
    if col3.button('Next'):
        st.session_state.menu = menu_options[7]
        st.rerun()
    
    elif col1.button('Previous'):
        st.session_state.menu = menu_options[5]
        st.rerun()


elif  st.session_state.menu== menu_options[7]:

    st.title('7-Renewable Generation')

    categories_solar_hotwater=X['SOLAR_WATER_HEATING_FLAG'].unique()
    st.session_state.selected_solar_hotwater=st.selectbox('Is the hot water in the property heated using a solar water heating system?', categories_solar_hotwater)
    
    st.session_state.max_pv=float(round(0.12*(st.session_state.selected_floor_area/2)*(1/0.819),1))
    st.session_state.installed_capacity_pv=st.number_input('Please enter installed photovoltaic capacity (KW)', min_value=0.00 , max_value=st.session_state.max_pv)
    pv_area=st.session_state.installed_capacity_pv*8.333
    st.session_state.roof_area=1
    if st.session_state.selected_roof_type=='Pitched Roof' and st.session_state.selected_property_type in ['House','Bungalow']:
         st.session_state.roof_area=(st.session_state.selected_floor_area/2)/0.819
    elif st.session_state.selected_roof_type=='Flat roof' and st.session_state.selected_property_type in ['House','Bungalow']: 
         st.session_state.roof_area=(st.session_state.selected_floor_area/2)
    else:
         pv_area=0
     
    st.session_state.selected_pv_supply=(pv_area/st.session_state.roof_area)*100


    col1, col2, col3=st.columns(3)
    if col1.button('Previous'):
        st.session_state.menu = menu_options[6]
        st.rerun()
    
    elif col3.button('Submit'):
        st.session_state.menu = menu_options[8]
        st.rerun()
        


elif  st.session_state.menu== menu_options[8]:

    st.title('Overview of the case study building')


    st.session_state.my_case_study = pd.DataFrame({
        'PROPERTY_TYPE': [st.session_state.selected_property_type],
        'BUILT_FORM': [st.session_state.selected_built_form],
        'TOTAL_FLOOR_AREA': [st.session_state.selected_floor_area],
        'MULTI_GLAZE_PROPORTION': [st.session_state.selected_multi_glazed_proportion],
        'GLAZED_TYPE': [st.session_state.selected_glazed_type],
        'GLAZED_AREA': [st.session_state.selected_glazed_area],
        'LOW_ENERGY_LIGHTING': [st.session_state.selected_low_energy_lighting],
        'HOTWATER_DESCRIPTION': [st.session_state.selected_hotwater],
        'SECONDHEAT_DESCRIPTION': [st.session_state.selected_secondary_heating],
        'FLOOR_HEIGHT': [st.session_state.selected_floor_height],
        'PHOTO_SUPPLY': [st.session_state.selected_pv_supply],
        'SOLAR_WATER_HEATING_FLAG': [st.session_state.selected_solar_hotwater],
        'MECHANICAL_VENTILATION': [st.session_state.selected_ventilation],
        'CONSTRUCTION_AGE_BAND': [st.session_state.selected_age_band],
        'HEATING_SYSTEM': [st.session_state.selected_heating_system],
        'FLOOR_TYPE': [st.session_state.selected_floor_type],
        'FLOOR_INSULATION': [st.session_state.selected_floor_insulation],
        'WALLS_U_VALUE': [st.session_state.selected_wall_u_value],
        'ROOF_TYPE': [st.session_state.selected_roof_type],
        'ROOF_INSULATION': [st.session_state.selected_roof_insulation],
    })

    #st.dataframe(st.session_state.my_case_study)

    st.session_state.my_case_study_table_summary1 = pd.DataFrame({
        'Type of Property': [st.session_state.selected_property_type],
        'Total Floor Area (sqm)': [st.session_state.selected_floor_area],
        'Glazed Area (sqm)': [st.session_state.glazing_area],
        'LOW_ENERGY_LIGHTING': [st.session_state.selected_low_energy_lighting],
        'Hotwater System': [st.session_state.selected_hotwater],
        'Floor to Ceiling Height': [st.session_state.selected_floor_height],
        'Main Heating System': [st.session_state.selected_heating_system],
        'Floor Insulation': [st.session_state.selected_floor_insulation],
        'Roof Insulation': [st.session_state.selected_roof_insulation],
    })

    st.session_state.my_case_study_table_summary2 = pd.DataFrame({
        'Built Form': [st.session_state.selected_built_form],
        'Glazed Type': [st.session_state.selected_glazed_type],
        'Share of energy efficient lighting (%)': [st.session_state.selected_low_energy_lighting],
        'Secondary Heating System': [st.session_state.selected_secondary_heating],
        'PV Capacity (Kw)': [st.session_state.installed_capacity_pv],
        'Mechanical Ventilation': [st.session_state.selected_ventilation],
        'Construction Age Band': [st.session_state.selected_age_band],
        'Floor Type': [st.session_state.selected_floor_type],
        'Walls U-Value': [st.session_state.selected_wall_u_value],
        'Roof Type': [st.session_state.selected_roof_type],
    })


    col1, col2 = st.columns(2)
    with col1:
        for key, value in st.session_state.my_case_study_table_summary1.iloc[0].items():
            st.markdown(f"**{key}** : {value}")

    with col2:
        for key, value in st.session_state.my_case_study_table_summary2.iloc[0].items():
            st.markdown(f"**{key}** : {value}")

    


    st.title('Energy Performance Assessment Results')


    mapping={'before 1900':0,'1900-1929':1,'1930-1949':2,'1950-1966':3,'1967-1975':4,'1976-1982':5,
        '1983-1990':6,'1991-1995':7,'1996-2002':8,'2003-2006':9,'2007-2011':10,'2012 onwards':11}
    X['CONSTRUCTION_AGE_BAND']=X['CONSTRUCTION_AGE_BAND'].map(mapping)
    st.session_state.my_case_study['CONSTRUCTION_AGE_BAND']=st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].map(mapping)


    mapping_2={'Single glazing':0,'Secondary glazing':1,'Double glazing':2,'Triple glazing':3}
    X['GLAZED_TYPE']=X['GLAZED_TYPE'].map(mapping_2)
    st.session_state.my_case_study['GLAZED_TYPE']=st.session_state.my_case_study['GLAZED_TYPE'].map(mapping_2)


    categorical_columns= ['PROPERTY_TYPE','BUILT_FORM', 'GLAZED_AREA','HOTWATER_DESCRIPTION','SECONDHEAT_DESCRIPTION',
                          'MECHANICAL_VENTILATION','HEATING_SYSTEM','SOLAR_WATER_HEATING_FLAG',
                          'FLOOR_TYPE','FLOOR_INSULATION','ROOF_TYPE','ROOF_INSULATION']  
    numerical_columns= ['TOTAL_FLOOR_AREA', 'MULTI_GLAZE_PROPORTION','LOW_ENERGY_LIGHTING','FLOOR_HEIGHT','PHOTO_SUPPLY','WALLS_U_VALUE','CONSTRUCTION_AGE_BAND','GLAZED_TYPE'] 
    
    st.write('')
    st.write('')
    st.write('')

    #scaling
    scaler=MinMaxScaler()
    X_scaled=X.copy()
    st.session_state.my_case_study_scaled=st.session_state.my_case_study.copy()
    X_scaled[numerical_columns]=scaler.fit_transform(X[numerical_columns])
    st.session_state.my_case_study_scaled[numerical_columns]=scaler.transform(st.session_state.my_case_study[numerical_columns])

    #encoding
    X_encoded_scaled=pd.get_dummies(X_scaled)
    st.session_state.my_case_study_encoded_scaled=pd.get_dummies(st.session_state.my_case_study_scaled).reindex(columns=X_encoded_scaled.columns, fill_value=0)
    encoder=LabelEncoder()
    Y_epc_encoded=encoder.fit_transform(Y_epc)

    mapping={0:'before 1900', 1:'1900-1929', 2:'1930-1949', 3:'1950-1966', 4:'1967-1975', 5:'1976-1982',
            6:'1983-1990', 7:'1991-1995', 8:'1996-2002', 9:'2003-2006', 10:'2007-2011', 11:'2012 onwards'}
    X['CONSTRUCTION_AGE_BAND']=X['CONSTRUCTION_AGE_BAND'].map(mapping)
    st.session_state.my_case_study['CONSTRUCTION_AGE_BAND']=st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].map(mapping)

    mapping_2={0:'Single glazing', 1:'Secondary glazing', 2:'Double glazing', 3:'Triple glazing'}
    X['GLAZED_TYPE']=X['GLAZED_TYPE'].map(mapping_2)
    st.session_state.my_case_study['GLAZED_TYPE']=st.session_state.my_case_study['GLAZED_TYPE'].map(mapping_2)



    #rule-based for PV calculation
    st.session_state.my_case_study_encoded_scaled_pv=st.session_state.my_case_study_encoded_scaled.copy()
    st.session_state.my_case_study_encoded_scaled_pv['PHOTO_SUPPLY']=0
    st.session_state.generated_electricity_pv=(st.session_state.installed_capacity_pv*24*365*0.11)/(st.session_state.selected_floor_area)

    #st.write(X_encoded_scaled)
    #st.write(my_case_study_encoded_scaled)
    #st.write(my_case_study_encoded_scaled_pv)


    X_encoded_scaled_train, X_encoded_scaled_test, Y_energy_train, Y_energy_test= train_test_split(X_encoded_scaled,Y_energy,test_size=0.15, random_state=100)
    X_encoded_scaled_train, X_encoded_scaled_test, Y_epc_encoded_train, Y_epc_encoded_test= train_test_split(X_encoded_scaled,Y_epc_encoded,test_size=0.15,random_state=100)
    X_encoded_scaled_train, X_encoded_scaled_test, Y_co2_train, Y_co2_test= train_test_split(X_encoded_scaled,Y_co2,test_size=0.15,random_state=100)
    
    #creating XGBoost model
    @st.cache_resource(ttl=24*3600)
    def XGBmodel(X_encoded_scaled_train,Y_energy_train):
        st.session_state.model=XGBRegressor()
        st.session_state.model.fit(X_encoded_scaled_train,Y_energy_train)
        return st.session_state.model
    
    @st.cache_resource(ttl=24*3600)
    def XGBmodel_epc(X_encoded_scaled_train,Y_epc_encoded_train):
         smote=SMOTE(random_state=42)
         X_resampled, Y_resampled=smote.fit_resample(X_encoded_scaled_train,Y_epc_encoded_train)
         #st.write(pd.Series(Y_resampled).value_counts())
         st.session_state.model_epc=XGBClassifier()
         st.session_state.model_epc.fit(X_resampled, Y_resampled)
         return st.session_state.model_epc
    
    @st.cache_resource(ttl=24*3600)
    def XGBmodel_co2(X_encoded_scaled_train,Y_co2_train):
         st.session_state.model_co2=XGBRegressor()
         st.session_state.model_co2.fit(X_encoded_scaled_train,Y_co2_train)
         return st.session_state.model_co2
      
    #cross val scores
    #model_temp=XGBClassifier()
    #smote=SMOTETomek(random_state=42)
    #pipeline = make_pipeline(smote, model_temp)
    #cv = KFold(n_splits=5, shuffle=True, random_state=42)
    #cv_score1 = cross_val_score(model_temp, X_encoded_scaled, Y_epc_encoded, cv=cv, scoring='accuracy')
    #st.write(cv_score1)
    #model_temp2=XGBRegressor()
    #cv_score3=cross_val_score(model_temp2,X_encoded_scaled,Y_energy,cv=5,scoring='r2')
    #cv_score4=cross_val_score(model_temp2,X_encoded_scaled,Y_energy,cv=5,scoring='neg_root_mean_squared_error')
    #cv_score5=cross_val_score(model_temp2,X_encoded_scaled,Y_co2,cv=5,scoring='r2')
    #cv_score6=cross_val_score(model_temp2,X_encoded_scaled,Y_co2,cv=5,scoring='neg_root_mean_squared_error')
    #st.write(cv_score3)
    #st.write(np.abs(cv_score4))
    #st.write(cv_score5)
    #st.write(np.abs(cv_score6))

    st.session_state.model=XGBmodel(X_encoded_scaled_train,Y_energy_train)
    st.session_state.model_epc=XGBmodel_epc(X_encoded_scaled_train,Y_epc_encoded_train)
    st.session_state.model_co2=XGBmodel_co2(X_encoded_scaled_train,Y_co2_train)


    #predictions
    st.session_state.y_predicted_energy=st.session_state.model.predict(st.session_state.my_case_study_encoded_scaled_pv)
    y_pred_XGB=st.session_state.model.predict(X_encoded_scaled_test)
    y_pred_epc=st.session_state.model_epc.predict(st.session_state.my_case_study_encoded_scaled)
    y_pred_co2=st.session_state.model_co2.predict(st.session_state.my_case_study_encoded_scaled)
    y_co2_accuracy=st.session_state.model_co2.predict(X_encoded_scaled_test)
    y_epc_accuracy=st.session_state.model_epc.predict(X_encoded_scaled_test)
    #r2 and accuracy scores
    r2_co2=r2_score(Y_co2_test,y_co2_accuracy)
    co2_error_percent=((y_co2_accuracy-Y_co2_test)/Y_co2_test)*100
    #plt.figure(figsize=(4,4))
    #sns.kdeplot(x=co2_error_percent, y=Y_co2_test, cmap="Greens", fill=True)
    #plt.xlabel('Error [%]')
    #plt.ylabel('CO2 emission [kg/m²]')
    #plt.xticks(ticks=[-100, -50, 0, 50, 100], labels=[-100, -50, 0, 50, 100])
    #st.pyplot(plt)

    r2_energy=r2_score(Y_energy_test,y_pred_XGB)
    energy_error_percent=((y_pred_XGB-Y_energy_test)/Y_energy_test)*100
    #plt.figure(figsize=(4,4))
    #sns.kdeplot(x=energy_error_percent, y=y_pred_XGB, cmap="Oranges", fill=True)
    #plt.xlabel('Error [%]')
    #plt.ylabel('Annual energy consumption [Kwh/m²]')
    #plt.xticks(ticks=[-100, -50, 0, 50, 100], labels=[-100, -50, 0, 50, 100])
    #st.pyplot(plt)

    accuracy_epc=accuracy_score(Y_epc_encoded_test,y_epc_accuracy)
    #accuracy_epc2=confusion_matrix(Y_epc_encoded_test,y_epc_accuracy)
    #disp=ConfusionMatrixDisplay(confusion_matrix=accuracy_epc2, display_labels=['B','C','D','E','F','G'])
    #fig,ax=plt.subplots()
    #disp.plot(ax=ax,cmap=plt.cm.Oranges)
    #st.pyplot(fig)
    

    #st.write(r2_co2)
    #st.write(r2_energy)
    #st.write(accuracy_epc)
    #st.write(accuracy_epc2)
  
    

    epc_mapping={
          0:'B',
          1:'C',
          2:'D',
          3:'E',
          4:'F',
          5:'G'
    }


    st.session_state['predicted_energy_consumption']=st.session_state.y_predicted_energy-st.session_state.generated_electricity_pv
    st.session_state['predicted_co2']=y_pred_co2
    st.session_state['predicted_epc']=epc_mapping[int(y_pred_epc)]




    #image_path="D:\PhD-UWL\codes\main\epc1.jpeg"
    i=1
    st.session_state['i']=i

    #triggering prediction
    col1, col2, col3=st.columns(3)
    col2.write(f"**CO2 emissions per square metre floor area per year is {int(y_pred_co2)}** (Kg/sqm)")
    col3.write(f"**Annual primary energy consumption is estimated {int(st.session_state.y_predicted_energy-st.session_state.generated_electricity_pv)}** (KWh/sqm)")
     
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
    

    st.write('')
    st.write('')
    st.write('')


    if st.button('Go to Retrofit Planning'):
        st.session_state.menu = menu_options[9]
        st.rerun()
    



###########RETROFIT#######################RETROFIT################RETROFIT#############RETROFIT###########RETROFIT#############RETROFIT################RETROFIT##############################

elif  st.session_state.menu== menu_options[9]:

     st.title('9-Retrofit Planning')

     

     y_predicted_epc=st.session_state['predicted_epc']
     y_predicted_co2=st.session_state['predicted_co2']

     st.session_state.my_case_study_retrofitted=st.session_state.my_case_study.copy()

     #st.write(my_case_study)
     


     #col1,col2=st.columns(2)
     #col1.write('__1-Externall wall retrofit__')

     #Retrofit prices    #Retrofit prices    #Retrofit prices               #Retrofit prices            #Retrofit prices             #Retrofit prices           #Retrofit prices
     st.session_state.wall_insulation_50mm=90            #GBP
     st.session_state.wall_insulation_100mm=110          #GBP
     st.session_state.wall_insulation_150mm=130          #GBP
     st.session_state.wall_insulation_200mm=150          #GBP
     cavity_filling=20                  #GBP
     double_glazed_window= 540          #GBP
     triple_glazed_window= 1200         #GBP
     suspended_floor_insulation=105     #GBP
     solid_floor_insulation= 80         #GBP
     roof_insulation_25mm=17.5          #GBP
     roof_insulation_50mm=20            #GBP
     roof_insulation_100mm=25           #GBP
     roof_insulation_200mm=35           #GBP
     fail=0

     #ASHP quote      #ASHP quote           #ASHP quote           #ASHP quote           #ASHP quote          #ASHP quote          #ASHP quote            #ASHP quote          #ASHP quote

     @st.cache_resource(ttl=24*3600)
     def quote_request(quote_features):
          #PATH = "chromedriver.exe"
          try:
               def installff():
                    os.system('sbase install geckodriver')
                    os.system('ln -s /home/appuser/venv/lib/python3.7/site-packages/seleniumbase/drivers/geckodriver /home/appuser/venv/bin/geckodriver')

               _ = installff()

               opts=FirefoxOptions()
               opts.add_argument("--headless")
               opts.add_argument("--disable-gpu")


               driver=webdriver.Firefox(options=opts)
               driver.get('https://www.edfenergy.com/heating/electric/air-source-heat-pump')

               accept_cookies = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
               accept_cookies.click()

               input_1=driver.find_element(By.XPATH,"//input[@id='edit-postcode']")
               input_1.send_keys("W55RF")

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit']")))
               submit.click()

               drop_down_1= Select(driver.find_element(By.XPATH,"(//select[@class='edf-standard-form-element border border-gray-500 pr-[50px]'])[1]"),)
               drop_down_1.select_by_visible_text("University Of West London, St Marys Road, Ealing")

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-own-home-yes']//div[@class='radio-btn-inner justify-center']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-outside-space-yes']//div[@class='radio-btn-inner justify-center']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-hot-water-cylinder-or-space-yes']//div[@class='radio-btn-inner justify-center']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-property-type-detached']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-how-many-floors-two']//div[@class='radio-btn-inner']")))
               submit.click()

               #floor area of property
               input_1=driver.find_element(By.XPATH,"//input[@id='edit-submitted-floor-area-floor-area-square-metres']")
               input_1.send_keys(quote_features['floor_area'])

               drop_down_2= Select(driver.find_element(By.XPATH,"//select[@id='edit-submitted-property-construction-year']"),)
               drop_down_2.select_by_visible_text("I am not sure")

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit--2']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-electricity-supply-one-phase']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit--2']")))
               submit.click()


               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-windows-double-triple-glazed-yes']//div[@class='radio-btn-inner justify-center']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-windows-glazed-before-2018-unknown']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-walls-type-unknown']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-walls-insulated-unknown']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-floor-insulated-unknown']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-roof-insulated-unknown']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-heating-type-gas']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-combi-boiler-yes']//div[@class='radio-btn-inner justify-center']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-room-heating-rads']//div[@class='radio-btn-inner']")))
               submit.click()

               #number of radiator
               input_2=driver.find_element(By.XPATH,"//input[@id='edit-submitted-no-of-radiators']")
               input_2.send_keys(quote_features['num_rooms'])

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@for='edit-submitted-radiator-pipes-copper-15mm-or-more']//div[@class='radio-btn-inner']")))
               submit.click()

               wait = WebDriverWait(driver, 10)
               submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@id='edit-submit']")))
               submit.click()

               cost_text=driver.find_element(By.XPATH,"//strong[@class='text-3xl']")
               #print(cost_text.text)
               cost_str=str(cost_text.text)
               cost_str=cost_str.replace(',','')
               cost_str=cost_str.replace('£','')
               cost_int=int(cost_str)+7500


               return cost_int
          
          except Exception:
               return "Quote request is not available right now! Please retry later."
     

        
    
 
     floor_area=str(st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])
     num_rooms=round((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/40)+2
     num_rooms=str(num_rooms)
     quote_features={
          'num_rooms': num_rooms,
          'floor_area': floor_area
     }

     ashp_price_quote=quote_request(quote_features)

     
     col1,col2=st.columns(2)
     col1.write('__1-Externall wall retrofit__')
     df_wall_u_value=pd.read_csv(r'wall-u-value.csv')
     df_wall_u_value.set_index('Unnamed: 0',inplace=True)


     if st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='As built':
     
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall', '100mm insulation for external wall', '150mm insulation for external wall',
                                                                           'No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_50mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 50mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='100mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='150mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_150mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 150mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='50-99mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('100mm insulation for external wall', '150mm insulation for external wall','No retrofit is required'))
          if wall_option=='100mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='150mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_150mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 150mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='100-149mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('150mm insulation for external wall','No retrofit is required'))
          if wall_option=='150mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_150mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['solid brick- 150mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Solid brick' and st.session_state.wall_insulation=='more than 150mm insulation':
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='As built':
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall','No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_50mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['timber frame- internal insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
     elif st.session_state.wall_type=='Timber frame' and st.session_state.wall_insulation=='Timber frame with internal insulation':
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='As built':
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall', '100mm insulation for external wall', 'Filling cavity'
                                                                           ,'Filling cavity and 50mm insulation','Filling cavity and 100mm insulation','No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_50mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['unfilled cavity- 50mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='100mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['unfilled cavity- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity':
               retrofit_cost=int(cavity_filling*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity and 50mm insulation':
               retrofit_cost=int(cavity_filling*st.session_state.external_wall_area)+int(st.session_state.wall_insulation_50mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 50mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity and 100mm insulation':
               retrofit_cost=int(cavity_filling*st.session_state.external_wall_area)+int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='50-99mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('100mm insulation for external wall', 'Filling cavity','Filling cavity and 100mm insulation','No retrofit is required'))
          if wall_option=='100mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['unfilled cavity- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity':
               retrofit_cost=int(cavity_filling*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 50mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='Filling cavity and 100mm insulation':
               retrofit_cost=int(cavity_filling*st.session_state.external_wall_area)+int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='more than 100mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('Filling cavity','No retrofit is required'))
          if wall_option=='Filling cavity':
               retrofit_cost=int(cavity_filling*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
     
     elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity':
          wall_option= col1.radio("Choose the external wall insulation:", ('50mm insulation for external wall', '100mm insulation for external wall','No retrofit is required'))
          if wall_option=='50mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_50mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 50mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='100mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with 50-99mm insulation':
          wall_option= col1.radio("Choose the external wall insulation:", ('100mm insulation for external wall','No retrofit is required'))
          if wall_option=='100mm insulation for external wall':
               retrofit_cost=int(st.session_state.wall_insulation_100mm*st.session_state.external_wall_area)
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
               st.session_state.my_case_study_retrofitted['WALLS_U_VALUE']=df_wall_u_value.loc['filled cavity- 100mm insulation',st.session_state.my_case_study['CONSTRUCTION_AGE_BAND'].iloc[0]]
          elif wall_option=='No retrofit is required':
               retrofit_cost=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')

     elif st.session_state.wall_type=='Cavity wall' and st.session_state.wall_insulation=='filled cavity with more than 100mm insulation':
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost} GBP**')
     

     col1,col2=st.columns(2)
     col1.write('__2-Retrofit glazing__')
     if st.session_state.my_case_study['GLAZED_TYPE'].iloc[0] in ['Secondary glazing','Single glazing']:
          glazing_option=col1.radio('Choose glazing type: ',('Double glazing','Triple glazing', 'No retrofit is required'))
          if glazing_option=='Double glazing':
               retrofit_cost_2=int(st.session_state.glazing_area*double_glazed_window)
               col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
               st.session_state.my_case_study_retrofitted['GLAZED_TYPE']='Double glazing'
          elif glazing_option=='Triple glazing':
               retrofit_cost_2=int(st.session_state.glazing_area*triple_glazed_window)
               col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
               st.session_state.my_case_study_retrofitted['GLAZED_TYPE']='Triple glazing'
          elif glazing_option=='No retrofit is required':
               retrofit_cost_2=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
     elif st.session_state.my_case_study['GLAZED_TYPE'].iloc[0] in ['Double glazing','Triple glazing']:
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_2=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_2} GBP**')
     
     col1,col2=st.columns(2)
     col1.write('__3-Floor retrofit__')
     if (st.session_state.my_case_study['FLOOR_TYPE'].iloc[0]=='Another dwelling or premises below') or (st.session_state.my_case_study['FLOOR_TYPE'].iloc[0] in 
                                                                                     ['Solid (next to the ground)','Suspended (next to the ground)','Exposed or to unheated space'] and 
                                                                                     st.session_state.my_case_study['FLOOR_INSULATION'].iloc[0]=='Insulated-at least 50mm insulation'):
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_3=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
     elif (st.session_state.my_case_study['FLOOR_TYPE'].iloc[0]=='Suspended (next to the ground)' and st.session_state.my_case_study['FLOOR_INSULATION'].iloc[0]=='As built'):
          floor_option= col1.radio('Choose floor insulation: ',['50mm insulation for floor','No retrofit is required'])
          if floor_option=='50mm insulation for floor':
               retrofit_cost_3=(st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])*suspended_floor_insulation
               col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
               st.session_state.my_case_study_retrofitted['FLOOR_INSULATION']='Insulated-at least 50mm insulation'
          elif floor_option=='No retrofit is required':
               retrofit_cost_3=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
     elif (st.session_state.my_case_study['FLOOR_TYPE'].iloc[0] in ['Solid (next to the ground)','Exposed or to unheated space'] and st.session_state.my_case_study['FLOOR_INSULATION'].iloc[0]=='As built'):
          floor_option= col1.radio('Choose floor insulation: ',['50mm insulation for floor','No retrofit is required'])
          if floor_option=='50mm insulation for floor':
               retrofit_cost_3=(st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])*solid_floor_insulation
               col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')
               st.session_state.my_case_study_retrofitted['FLOOR_INSULATION']='Insulated-at least 50mm insulation'
          elif floor_option=='No retrofit is required':
               retrofit_cost_3=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_3} GBP**')


     col1,col2=st.columns(2)
     col1.write('__4-Roof retrofit__')
     if (st.session_state.my_case_study['ROOF_TYPE'].iloc[0]=='Another dwelling or premises above') or (st.session_state.my_case_study['ROOF_TYPE'].iloc[0] in ['Flat roof', 'Roof room'] and 
                                                                                     st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='Insulated-unknown thickness (50mm or more)') or (st.session_state.my_case_study['ROOF_TYPE'].iloc[0] =='Pitched Roof' and st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='More than 200mm loft insulation'):
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_4=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (st.session_state.my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='As built'):
          roof_option=col1.radio('Choose roof insulation: ', ['25mm loft insulation','50mm loft insulation','100mm loft insulation','200mm loft insulation','No retrofit is required'])
          if roof_option=='25mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_25mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='less than 50mm loft insulation'
          elif roof_option=='50mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='50 to 99mm loft insulation'
          elif roof_option=='100mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_100mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='100 to 200mm loft insulation'
          elif roof_option=='200mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (st.session_state.my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='less than 50mm loft insulation'):
          roof_option=col1.radio('Choose roof insulation: ', ['50mm loft insulation','100mm loft insulation','200mm loft insulation','No retrofit is required'])
          if roof_option=='50mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='50 to 99mm loft insulation'
          elif roof_option=='100mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_100mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='100 to 200mm loft insulation'
          elif roof_option=='200mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (st.session_state.my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='50 to 99mm loft insulation'):
          roof_option=col1.radio('Choose roof insulation: ', ['100mm loft insulation','200mm loft insulation','No retrofit is required'])
          if roof_option=='100mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_100mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='100 to 200mm loft insulation'
          elif roof_option=='200mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (st.session_state.my_case_study['ROOF_TYPE'].iloc[0]=='Pitched Roof') and (st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='100 to 200mm loft insulation'):
          roof_option=col1.radio('Choose roof insulation: ', ['200mm loft insulation','No retrofit is required'])
          if roof_option=='200mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_200mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='More than 200mm loft insulation'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (st.session_state.my_case_study['ROOF_TYPE'].iloc[0]=='Flat roof') and (st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='As built'):
          roof_option=col1.radio('Choose roof insulation: ', ['50mm insulation','No retrofit is required'])
          if roof_option=='50mm loft insulation':
               retrofit_cost_4=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='Insulated-unknown thickness (50mm or more)'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')
     elif (st.session_state.my_case_study['ROOF_TYPE'].iloc[0]=='Roof room') and (st.session_state.my_case_study['ROOF_INSULATION'].iloc[0]=='As built'):
          roof_option=col1.radio('Choose roof insulation: ', ['50mm loft insulation','No retrofit is required'])
          if roof_option=='50mm loft insulation':
               roof_room_floor=((st.session_state.my_case_study['TOTAL_FLOOR_AREA'].iloc[0])/2)
               roof_room_wall=((roof_room_floor*0.3*0.66)**0.5)*8.25
               retrofit_cost_4=(roof_room_floor+roof_room_wall)*roof_insulation_50mm
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4:.2f} GBP**')
               st.session_state.my_case_study_retrofitted['ROOF_INSULATION']='Insulated-unknown thickness (50mm or more)'
          elif roof_option=='No retrofit is required':
               retrofit_cost_4=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_4} GBP**')


     col1,col2=st.columns(2)
     col1.write('__5-Heating system retrofit__')
     if st.session_state.my_case_study['HEATING_SYSTEM'].iloc[0] in ['Boiler system with radiators or underfloor heating','Electric storage system','Electric underfloor heating',
                                                  'Room heater','Warm air system (not heat pump)']:
          heating_option=col1.radio('Choose heating system retrofit: ',['Air source heat pump','No retrofit is required'])
          if heating_option=='Air source heat pump':
               col2.write(f'Air source heat pump cost (assuming no need of radiator upgrades): {int(ashp_price_quote)} GBP')
               gov_grant=7500
               col2.write(f'Boiler upgrade scheme government grant: {gov_grant} GBP')
               retrofit_cost_5=ashp_price_quote-gov_grant
               col2.write(f'**Estimated total cost after government grant is {int(retrofit_cost_5)} GBP**')
               st.session_state.my_case_study_retrofitted['HEATING_SYSTEM'] = 'Air source heat pump with radiators or underfloor heating'
               #st.session_state.my_case_study_retrofitted['MAIN_FUEL']='All electric'
               
          elif heating_option=='No retrofit is required':
               retrofit_cost_5=0
               col2.write(f'**estimated retrofit cost is {retrofit_cost_5} GBP**')

     elif st.session_state.my_case_study ['HEATING_SYSTEM'].iloc[0] in ['Air source heat pump with radiators or underfloor heating','Air source heat pump with warm air distribution']:
          col1.write('')
          col1.write('No retrofit is required')
          col1.write('')
          retrofit_cost_5=0
          col2.write(f'**estimated retrofit cost is {retrofit_cost_5} GBP**')
     

     st.write('')
     col1,col2=st.columns(2)
     col1.write('__6-Adding PV solar to the property__')
     pv_retrofit=col1.number_input('Please enter required PV capacity (KWp) :', min_value=0.00 , max_value=st.session_state.max_pv-st.session_state.installed_capacity_pv)
     pv_area_retrofit=pv_retrofit*8.333
     st.session_state.my_case_study_retrofitted['PHOTO_SUPPLY']=((pv_area_retrofit/st.session_state.roof_area)*100) + st.session_state.my_case_study['PHOTO_SUPPLY'].iloc[0]
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




     categorical_columns= ['PROPERTY_TYPE','BUILT_FORM', 'GLAZED_AREA','HOTWATER_DESCRIPTION','SECONDHEAT_DESCRIPTION',
                              'MECHANICAL_VENTILATION','HEATING_SYSTEM','SOLAR_WATER_HEATING_FLAG',
                              'FLOOR_TYPE','FLOOR_INSULATION','ROOF_TYPE','ROOF_INSULATION']  
     numerical_columns= ['TOTAL_FLOOR_AREA', 'MULTI_GLAZE_PROPORTION','LOW_ENERGY_LIGHTING','FLOOR_HEIGHT','PHOTO_SUPPLY','WALLS_U_VALUE','CONSTRUCTION_AGE_BAND','GLAZED_TYPE']

     mapping_3={'before 1900':0,'1900-1929':1,'1930-1949':2,'1950-1966':3,'1967-1975':4,'1976-1982':5,
          '1983-1990':6,'1991-1995':7,'1996-2002':8,'2003-2006':9,'2007-2011':10,'2012 onwards':11}
     
     mapping_4={'Single glazing':0,'Secondary glazing':1,'Double glazing':2,'Triple glazing':3}

     st.session_state.my_case_study_retrofitted['CONSTRUCTION_AGE_BAND']=st.session_state.my_case_study_retrofitted['CONSTRUCTION_AGE_BAND'].map(mapping_3)
     st.session_state.my_case_study_retrofitted['GLAZED_TYPE']=st.session_state.my_case_study_retrofitted['GLAZED_TYPE'].map(mapping_4)
     



     scaler=MinMaxScaler()
     X_scaled=X.copy()
     X['CONSTRUCTION_AGE_BAND']=X['CONSTRUCTION_AGE_BAND'].map(mapping_3)
     X['GLAZED_TYPE']=X['GLAZED_TYPE'].map(mapping_4)
     X_scaled[numerical_columns]=scaler.fit_transform(X[numerical_columns])
     st.session_state.my_case_study_retrofitted_scaled=st.session_state.my_case_study_retrofitted.copy()
     st.session_state.my_case_study_retrofitted_scaled[numerical_columns]=scaler.transform(st.session_state.my_case_study_retrofitted[numerical_columns])

     st.session_state.my_case_study_retrofitted_encoded_scaled=pd.get_dummies(st.session_state.my_case_study_retrofitted_scaled).reindex( columns=st.session_state.my_case_study_encoded_scaled.columns , fill_value=0)


     


     #rule based for PV
     st.session_state.my_case_study_retrofitted_encoded_scaled_pv=st.session_state.my_case_study_retrofitted_encoded_scaled.copy()
     st.session_state.my_case_study_retrofitted_encoded_scaled_pv['PHOTO_SUPPLY']=0
     total_pv_retrofitted=pv_retrofit+st.session_state.installed_capacity_pv
     generated_pv_retrofitted=(total_pv_retrofitted*24*365*0.11)/st.session_state.selected_floor_area



     y_pred_epc_retrofitted=st.session_state.model_epc.predict(st.session_state.my_case_study_retrofitted_encoded_scaled)
     y_pred_energy_retrofitted=st.session_state.model.predict(st.session_state.my_case_study_retrofitted_encoded_scaled_pv)
     y_pred_co2_retrofitted=st.session_state.model_co2.predict(st.session_state.my_case_study_retrofitted_encoded_scaled)

     


     epc_mapping={
          0:'B',
          1:'C',
          2:'D',
          3:'E',
          4:'F',
          5:'G'
     }


     #st.session_state.my_case_study_retrofitted_encoded_scaled

     #st.session_state.my_case_study_encoded_scaled

     #st.session_state.my_case_study_retrofitted_scaled

     #st.session_state.my_case_study_scaled


     
     col1, col2, col3= st.columns(3)
     
     col3.write(f"annual energy consumption before retrofit was **{int(st.session_state.y_predicted_energy-st.session_state.generated_electricity_pv)}** and after retrofit is estimated **{int(y_pred_energy_retrofitted-generated_pv_retrofitted)}** (KWh/sqm)")

     col2.write(f"CO2 emissions per square metre floor area per year before retrofit was **{int(y_predicted_co2)}** and after retrofit is estimated **{int(y_pred_co2_retrofitted)}** (Kg/sqm)")


     col1.write(f'EPC rating before retrofit was **{y_predicted_epc}** and after retrofit is **{epc_mapping[int(y_pred_epc_retrofitted)]}**')    
    

    


    



          


