import streamlit as st
from streamlit_option_menu import option_menu

#from animation import*
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


#page congiguration
st.set_page_config(page_title= "Copper Modelling",
                   page_icon= 'random',
                   layout= "wide",)


#=========hide the streamlit main and footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: red;'>COPPER SELLING PRICE PREDICTION AND STATUS</h1>",
                unsafe_allow_html=True)

selected = option_menu(None, ["PREDICT SELLING PRICE","PREDICT STATUS",],
                           icons=['cash-coin','trophy'],orientation='horizontal',default_index=0)

if selected == 'PREDICT SELLING PRICE':
    status_value = ['Won','Revised','Lost','Not lost for AM','Draft','To be approved' ,'Offerable','Offered','Wonderful']
    item_type_value = ['S','W','WI','PL','Others','IPL','SLAWR']
    country_value = ['27.', '40.','26.','32.','78.','84.','25.','77.','30.','28.','38.','39.','107.','79.','80.','113.','89.']
    application_value = ['59.','10.','20.','29.','25.','42.','28.','41.','15.','27.','56.','39.','67.','22.','79.','40.',' 4.',' 3.','26.','38.','99.','69.','5.','58.','68.','66.','65.','70.','19.',' 2.']
    product_value = ['1670798778','640405','164141591','628377','611993','611728'
                    ,'628112','640665','1668701698','1671863738','1693867550','1332077137'
                    ,'1693867563','1668701718','1668701376','1671876026','640400','1665572374'
                    ,'1665572032','628117','164336407','164337175','1282007633','1690738206'
                    ,'1668701725','1665584642','611733','1722207579','1721130331','929423819'
                    ,'1690738219','1665584320','1665584662']
    c1,c2,c3=st.columns([2,2,2])
    with c1:
        quantity=st.text_input('Enter Quantity in tons')
        thickness = st.text_input('Enter Thickness')
        width = st.text_input('Enter Width ')
    with c2:
        country = st.selectbox('Country Code', country_value)
        status = st.selectbox('Status', status_value)
        item = st.selectbox('Item Type', item_type_value)
    with c3:
        application = st.selectbox('Application Type', application_value)
        product = st.selectbox('Product Reference', product_value)
    with c1:
        st.write('')
        st.write('')
        st.write('')
        if st.button('PREDICT SELLING PRICE'):
            with open('status.pkl', 'rb') as file:
                encoded_status = pickle.load(file)
            with open('item_type.pkl', 'rb') as file:
                encoded_item_type = pickle.load(file)
            with open('scaler.pkl', 'rb') as file:
                scaled_data = pickle.load(file)
            with open('regression.pkl','rb') as file:
                reg = pickle.load(file)
            encode_st = None
            for i, j in zip(status_value, encoded_status):
                if status == i:
                    encode_st = j
                    break
            else:
                st.error("Status not found.")
                exit()
            
            encode_it = None
            for i, j in zip(item_type_value, encoded_item_type):
                if item == i:
                    encode_it = j
                    break
            else:
                st.error("Item type not found.")
                exit()
            data =[]
            data.append(quantity)
            data.append(encode_st)
            data.append(encode_it)
            data.append(application)
            data.append(thickness)
            data.append(width)
            data.append(country)
            data.append(product)
            x = np.array(data).reshape(1, -1)
            pred_model = scaled_data.transform(x)
            price_predict= reg.predict(pred_model)
            predicted_price = str(price_predict)[1:-1]
            st.write(f'Predicted Selling Price : :green[₹] :green[{predicted_price}]')
            
if selected=='PREDICT STATUS':

        status_value_cls = ['Won','Revised','Lost','Not lost for AM','Draft','To be approved' ,'Offerable','Offered','Wonderful']
        item_type_value_cls = ['S','W','WI','PL','Others','IPL','SLAWR']
        country_value_cls = ['27.', '40.','26.','32.','78.','84.','25.','77.','30.','28.','38.','39.','107.','79.','80.','113.','89.']
        application_value_cls = ['59.','10.','20.','29.','25.','42.','28.','41.','15.','27.','56.','39.','67.','22.','79.','40.',' 4.',' 3.','26.','38.','99.','69.','5.','58.','68.','66.','65.','70.','19.',' 2.']
        product_value_cls = ['1670798778','640405','164141591','628377','611993','611728'
                        ,'628112','640665','1668701698','1671863738','1693867550','1332077137'
                        ,'1693867563','1668701718','1668701376','1671876026','640400','1665572374'
                        ,'1665572032','628117','164336407','164337175','1282007633','1690738206'
                        ,'1668701725','1665584642','611733','1722207579','1721130331','929423819'
                        ,'1690738219','1665584320','1665584662']

        cc1, cc2, cc3 = st.columns([2, 2, 2])
        with cc1:
            quantity_cls = st.text_input('Enter Quantity in tons')
            thickness_cls = st.text_input('Enter Thickness')
            width_cls= st.text_input('Enter Width')

        with cc2:
            selling_price_cls= st.text_input('Enter Selling Price')
            item_cls = st.selectbox('Item Type', item_type_value_cls)
            country_cls= st.selectbox('Country Code', country_value_cls)

        with cc3:
            application_cls = st.selectbox('Application Type', application_value_cls)
            product_cls = st.selectbox('Product Reference', product_value_cls)
            
        with cc1:
            st.write('')
            st.write('')
            st.write('')
            if st.button('PREDICT STATUS'):
                data = []
                with open('status.pkl', 'rb') as file:
                    encoded_status = pickle.load(file)
                with open('item_type.pkl', 'rb') as file:
                    encoded_item_type = pickle.load(file)
                with open('scaling_classify.pkl', 'rb') as file:
                    scaled_data_cls = pickle.load(file)
                with open('Decission_tree_classification.pkl', 'rb') as file:
                    trained_model_cls = pickle.load(file)

                encode_it_cls = None
                for i, j in zip(item_type_value_cls, encoded_item_type):
                    if item_cls == i:
                        encode_it_cls = j
                        break
                else:
                    st.error("Item type not found.")
                    exit()
                data.append(quantity_cls)
                data.append(selling_price_cls)
                data.append(encode_it_cls)
                data.append(application_cls)
                data.append(thickness_cls)
                data.append(width_cls)
                data.append(country_cls)
                data.append(product_cls)

                x_cls = np.array(data).reshape(1, -1)
                st.write(x_cls)
                scaling_model_cls = scaled_data_cls.transform(x_cls)
                pred_status = trained_model_cls.predict(scaling_model_cls)
                if pred_status==6:
                    st.write(f'Predicted Status : :green[WON]')
                else:
                    st.write(f'Predicted Status : :red[LOST]')

        st.info("The Predicted Status may be differ from various reason like Supply and Demand Imbalances,Infrastructure and Transportation etc..",icon='ℹ️')
