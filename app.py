
import streamlit as st

# Page setup
st.set_page_config(page_title="Agritech Pest Analysis", layout="wide")
st.title("🌾 Agritech Pest Analysis Dashboard")

# Sidebar options
option = st.sidebar.selectbox(
    'Select a module:',
    ('EDA & Preprocessing', 
     'Classification', 
     'Prediction (Linear)', 
     'Prediction (ARIMA)', 
     'Prediction (SARIMAX)', 
     'Prediction (SARIMA)', 
     'Prediction (ARIMAX)')
)

# Run modules based on user selection
if option == 'EDA & Preprocessing':
    import Agritech_Pest_EDA_and_Preprocessing as eda
    eda.main()

elif option == 'Classification':
    import Agritech_Pest_Classification as classify
    classify.main()

elif option == 'Prediction (Linear)':
    import Agritech_Pest_Prediction_LinearProblem as linear
    linear.main()

elif option == 'Prediction (ARIMA)':
    import Agritech_Pest_Prediction_ARIMA_TSA as arima
    arima.main()

elif option == 'Prediction (SARIMAX)':
    import Agritech_Pest_Prediction_SARIMAX_TSA as sarimax
    sarimax.main()

elif option == 'Prediction (SARIMA)':
    import Agritech_Pest_Prediction_SARIMA_TSA as sarima
    sarima.main()

elif option == 'Prediction (ARIMAX)':
    import Agritech_Pest_Prediction_ARIMAX_TSA as arimax
    arimax.main()
