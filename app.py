import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Restaurant Success Predictor - Bangalore",
    page_icon="🍽️",
    layout="wide"
)

# Cache data loading to improve performance
@st.cache_data
def load_data():
    try:
        return pd.read_csv("zomato_cleaned.csv")
    except FileNotFoundError:
        st.error("Dataset file not found. Please check if 'zomato_cleaned.csv' exists in the current directory.")
        return None

# Cache model loading to improve performance
@st.cache_resource
def load_models():
    models_dir = Path("models")
    try:
        return {
            "model": joblib.load(models_dir / 'model.h5'),
            "scaler": joblib.load(models_dir / 'scaler.h5'),
            "listed_in_city_encoder": joblib.load(models_dir / 'listed_in_city_Encoder.h5'),
            "listed_in_type_encoder": joblib.load(models_dir / 'listed_in_type_Encoder.h5'),
            "location_encoder": joblib.load(models_dir / 'location_Encoder.h5')
        }
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    # Load data and models
    df = load_data()
    if df is None:
        return
    
    models = load_models()
    if models is None:
        return
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5722;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #666;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>Restaurant Success Predictor - Bangalore</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>This tool uses machine learning to predict whether your new restaurant in Bangalore will be successful based on various factors.</p>", unsafe_allow_html=True)
    
    # Create a form to collect all inputs before prediction
    with st.form("prediction_form"):
        st.markdown("<h2 class='sub-header'>Enter Restaurant Details</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Left column inputs
        with col1:
            try:
                price_for_two = st.number_input(
                    'Approximate price for two (₹)',
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=50,
                    help="Enter the average cost for two people in rupees"
                )
                
                location = st.selectbox(
                    'Restaurant Location',
                    options=sorted(df['location'].unique()),
                    help="Select the location of your restaurant in Bangalore"
                )
                
                rest_type = st.multiselect(
                    'Restaurant Type',
                    options=sorted(list(df.columns)[8:32]),
                    help="Select the type(s) of your restaurant (e.g., Casual Dining, Café)"
                )
                
                online_ordering = st.checkbox(
                    "Will support online ordering", 
                    value=True,
                    help="Check if your restaurant will offer online ordering services"
                )
            except Exception as e:
                st.error(f"Error in left column: {e}")
                return
        
        # Right column inputs
        with col2:
            try:
                listed_in_city = st.selectbox(
                    'Listed in City',
                    options=sorted(df['listed_in(city)'].unique()),
                    help="Select the city area where your restaurant will be listed"
                )
                
                listed_in_type = st.selectbox(
                    'Listed in Type', 
                    options=sorted(df['listed_in(type)'].unique()),
                    help="Select the type category where your restaurant will be listed"
                )
                
                cuisines = st.multiselect(
                    'Cuisines Offered',
                    options=sorted(list(df.columns)[32:]),
                    help="Select the cuisines your restaurant will offer"
                )
                
                table_booking = st.checkbox(
                    "Will support table booking", 
                    value=True,
                    help="Check if your restaurant will offer table booking services"
                )
            except Exception as e:
                st.error(f"Error in right column: {e}")
                return
            
        # Submit button
        submit_button = st.form_submit_button("Predict Success")
    
    # Make prediction when the form is submitted
    if submit_button:
        try:
            with st.spinner("Analyzing restaurant details..."):
                # Input validation
                if not cuisines:
                    st.warning("Please select at least one cuisine.")
                    return
                
                if not rest_type:
                    st.warning("Please select at least one restaurant type.")
                    return
                
                # Prepare input data
                inp_data = []
                inp_data.append(1 if online_ordering else 0)
                inp_data.append(1 if table_booking else 0)
                inp_data.append(int(models["location_encoder"].transform([location])[0]))
                inp_data.append(int(price_for_two))
                inp_data.append(int(models["listed_in_type_encoder"].transform([listed_in_type])[0]))
                inp_data.append(int(models["listed_in_city_encoder"].transform([listed_in_city])[0]))
                inp_data.append(len(cuisines))
                
                # Add one-hot encoded restaurant types
                for col in list(df.columns)[8:32]:
                    inp_data.append(1 if col in rest_type else 0)
                
                # Add one-hot encoded cuisines
                for col in list(df.columns)[32:]:
                    inp_data.append(1 if col in cuisines else 0)
                
                # Transform and predict
                input_scaled = models["scaler"].transform([inp_data])
                prediction = models["model"].predict(input_scaled)[0]
                
                # Display prediction
                st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
                
                if prediction == "Yes":
                    st.success("🌟 Congratulations! Your restaurant is predicted to succeed! 🌟")
                    
                    # Show restaurant summary
                    st.markdown("### Restaurant Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Location:** {location}")
                        st.markdown(f"**Price for Two:** ₹{price_for_two}")
                        st.markdown(f"**Online Ordering:** {'Yes' if online_ordering else 'No'}")
                        st.markdown(f"**Table Booking:** {'Yes' if table_booking else 'No'}")
                    with col2:
                        st.markdown(f"**Restaurant Types:** {', '.join(rest_type)}")
                        st.markdown(f"**Cuisines:** {', '.join(cuisines)}")
                        st.markdown(f"**Listed in City:** {listed_in_city}")
                        st.markdown(f"**Listed in Type:** {listed_in_type}")
                    
                    # Success factors
                    st.markdown("### Key Success Factors")
                    success_factors = []
                    if online_ordering:
                        success_factors.append("Online ordering availability")
                    if price_for_two < 800:
                        success_factors.append("Competitive pricing")
                    if len(cuisines) >= 3:
                        success_factors.append("Diverse cuisine offerings")
                    
                    for factor in success_factors:
                        st.markdown(f"- {factor}")
                    
                else:
                    st.error("⚠️ Analysis suggests your restaurant may face challenges ⚠️")
                    
                    # Improvement suggestions
                    st.markdown("### Suggested Improvements")
                    suggestions = []
                    if not online_ordering:
                        suggestions.append("Consider adding online ordering services")
                    if price_for_two > 1000:
                        suggestions.append("Review your pricing strategy, consider more affordable options")
                    if len(cuisines) < 3:
                        suggestions.append("Expand your cuisine offerings to attract a wider audience")
                    if not table_booking:
                        suggestions.append("Add table booking services to improve customer experience")
                    
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    # Add helpful information in an expandable section
    with st.expander("ℹ️ About This Predictor"):
        st.markdown("""
        ### How it works
        This predictor uses machine learning to analyze patterns from existing Bangalore restaurants to determine success factors.
        
        ### Factors considered:
        * Restaurant location and type
        * Cuisines offered
        * Price point
        * Online ordering and table booking capabilities
        * And many more underlying features
        
        ### Limitations
        This is a prediction based on historical data and may not account for all market conditions or unique restaurant features.
        """)
    
    # Footer
    st.markdown("<div class='footer'>Powered by Machine Learning | Developed using Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
