import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from disease_preprocess import count_cases
from disease_outbreak import detect_outbreak_per_day, predict_future_outbreaks
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pydeck as pdk


model = joblib.load('models/disease_prediction_model.pkl')
le = joblib.load('encoders/label_encoder.pkl')




symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
    'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

location = ['Pasig City', 'Marikina City', 'Quezon City']


def symptom_checker():
    st.subheader("🩺 Disease Prediction")

    
    if 'predicted_disease' not in st.session_state:
        st.session_state.predicted_disease = None
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = {}

    patient_name = st.text_input("📝 **Patient Name**")
    patient_age = st.number_input("🎂 **Age**", min_value=0, max_value=120, step=1)
    patient_gender = st.selectbox("⚧️ **Gender**", ["Male", "Female"], index=None)
    locations = ['Pasig City', 'Marikina City', 'Quezon City']
    patient_location = st.selectbox('🌍 **Location**', locations, index=None)

   
    symptom_map = {s.replace("_", " ").replace("  ", " ").strip().capitalize(): s for s in symptoms}
    display_symptoms = list(symptom_map.keys())
    selected_symptoms_display = st.multiselect("🦠 **Select Symptoms**", display_symptoms)

    predict_button = st.button("🔍 **Predict Disease**")

    if predict_button:
        if not selected_symptoms_display:
            st.warning("⚠️ **Please select at least one symptom.**")
            return
        if not patient_name or not patient_location or patient_gender is None:
            st.warning("⚠️ **Please fill in all patient details.**")
            return

       
        selected_symptoms = [symptom_map[disp] for disp in selected_symptoms_display]
        user_input = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in symptoms}
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        predicted_disease = le.inverse_transform(prediction)[0]

     
        st.session_state.predicted_disease = predicted_disease
        st.session_state.selected_symptoms = selected_symptoms
        st.session_state.patient_info = {
            'Name': patient_name,
            'Age': patient_age,
            'Gender': patient_gender,
            'Location': patient_location
        }

       
        st.markdown(f"""
        ### ✨ **Patient Details:**
        - 👤 **Name:** {patient_name}
        - 🎂 **Age:** {patient_age}
        - ⚧️ **Gender:** {patient_gender}
        - 🌍 **Location:** {patient_location}

        ### 🔬 **Predicted Disease:**  
        🦠 **{predicted_disease}**
        """)

    
    if st.session_state.predicted_disease:
        if st.button("💾 **Record Patient Data**"):
          
            record = {
                'Date': pd.to_datetime('today').normalize().date(),
                'Name': st.session_state.patient_info.get('Name', ''),
                'Age': st.session_state.patient_info.get('Age', ''),
                'Gender': st.session_state.patient_info.get('Gender', ''),
                'Location': st.session_state.patient_info.get('Location', ''),
                'Disease': st.session_state.predicted_disease,
            }
        
            for i in range(1, 18):
                record[f'Symptom_{i}'] = st.session_state.selected_symptoms[i - 1] if i <= len(st.session_state.selected_symptoms) else ''

        
            csv_path = 'dataset/sample_user_data.csv'
            file_exists = os.path.isfile(csv_path)
            record_df = pd.DataFrame([record])
            record_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

            st.success("✅ **Patient information and disease record saved successfully!**")

        
            with open(csv_path, 'rb') as f:
                st.download_button(label="📥 **Download CSV**", data=f, file_name='patient_records.csv', mime='text/csv')

           
            st.session_state.predicted_disease = None
            st.session_state.selected_symptoms = []
            st.session_state.patient_info = {}

        

 

def outbreak_forecasting():
    st.subheader("Outbreak Detection and Forecasting")
    if st.button("Detect and Forecast Outbreaks"):
        try:
          
            df = pd.read_csv('dataset/sample_user_data.csv')
            if 'Date' not in df.columns or 'Disease' not in df.columns:
                st.error("Required columns 'Date' and 'Disease' missing in dataset.")
                return

          
            disease_counts = count_cases(df)
            detected_outbreaks = detect_outbreak_per_day(disease_counts)
            forecasted_outbreaks = predict_future_outbreaks(disease_counts, days=7)

            st.write("Detected Outbreaks:", detected_outbreaks)
            st.write("Forecasted Outbreaks:", forecasted_outbreaks)

         
            split_date = detected_outbreaks['Date'].max()
            forecasted_outbreaks['source'] = forecasted_outbreaks['Date'].apply(
                lambda d: 'historical' if pd.to_datetime(d) <= pd.to_datetime(split_date) else 'forecasted'
            )

       
            for disease in forecasted_outbreaks['Disease'].unique():
                disease_data = forecasted_outbreaks[forecasted_outbreaks['Disease'] == disease]
                fig, ax = plt.subplots(figsize=(10, 5))
                hist = disease_data[disease_data['source'] == 'historical']
                ax.plot(hist['Date'], hist['Cases'], label='Historical', marker='o', color='blue')
                forecast = disease_data[disease_data['source'] == 'forecasted']
                if not forecast.empty:
                    ax.plot(forecast['Date'], forecast['Cases'], label='Forecasted', marker='o', color='orange')

                outbreak_dates = disease_data[disease_data['Outbreak'] == 1]['Date']
                outbreak_cases = disease_data[disease_data['Outbreak'] == 1]['Cases']
                ax.scatter(outbreak_dates, outbreak_cases, color='red', label='Outbreak', zorder=5)
                ax.set_title(f'Disease Cases and Outbreaks: {disease}')
                ax.set_xlabel('Date')
                fig.autofmt_xdate() 
                ax.set_ylabel('Cases')
                ax.legend()
                st.pyplot(fig)

        
                outbreak_rows = disease_data[disease_data['Outbreak'] == 1]
                latest_date = disease_data['Date'].max()
                if latest_date not in outbreak_rows['Date'].values:
                    latest_row = disease_data[disease_data['Date'] == latest_date]
                    outbreak_rows = pd.concat([outbreak_rows, latest_row])

                if not outbreak_rows.empty:
                    st.markdown(f"**User details for {disease} Outbreaks:**")
                    for _, row in outbreak_rows.iterrows():
                        outbreak_date = row['Date']
                        matching_users = df[
                            (pd.to_datetime(df['Date']) == pd.to_datetime(outbreak_date)) &
                            (df['Disease'] == disease)
                        ][['Name', 'Age', 'Gender', 'Location']]
                        if not matching_users.empty:
                            num_cases = len(matching_users)
                            with st.expander(f"Outbreak on {outbreak_date} ({num_cases} case{'s' if num_cases != 1 else ''})"):
                                st.dataframe(matching_users.reset_index(drop=True))

   
            summary = df.groupby(['Location', 'Disease']).size().reset_index(name='Cases')
            max_row = summary.loc[summary['Cases'].idxmax()]
            max_cases = max_row['Cases']
            max_location = max_row['Location']
            max_disease = max_row['Disease']

            threshold = 5
            if max_cases > threshold:
                st.warning(f"🚨 **Possible Outbreak detected:** {max_location} → {max_disease} ({int(max_cases)} cases)")
            else:
                st.info("No significant outbreaks detected based on current data.")

           
            st.subheader("🗺️ Heatmap of Recorded Disease Cases")
            city_coords = {
                'Pasig City': {'lat': 14.5764, 'lon': 121.0851},
                'Marikina City': {'lat': 14.6507, 'lon': 121.1029},
                'Quezon City': {'lat': 14.6760, 'lon': 121.0437}
            }
            df['latitude'] = df['Location'].map(lambda x: city_coords.get(x, {}).get('lat'))
            df['longitude'] = df['Location'].map(lambda x: city_coords.get(x, {}).get('lon'))
            df_heat = df.dropna(subset=['latitude', 'longitude'])

            if not df_heat.empty:
                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=df_heat['latitude'].mean(),
                        longitude=df_heat['longitude'].mean(),
                        zoom=11,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'HeatmapLayer',
                            data=df_heat,
                            get_position='[longitude, latitude]',
                            get_weight=1,
                            radiusPixels=60,
                        )
                    ],
                ))
            else:
                st.warning("No latitude/longitude data available for heatmap.")

        except Exception as e:
            st.error(f"Outbreak Detection failed: {e}")



def main():
    tab1, tab2 = st.tabs(["🩺 Disease Prediction", "📊 Outbreak Forecasting"])
    with tab1:
        symptom_checker()
    with tab2:
        outbreak_forecasting()


if __name__ == "__main__":
    main()
