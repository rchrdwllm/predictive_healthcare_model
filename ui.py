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


model = joblib.load('models/disease_prediction_model.joblib')
symptoms = ['muscle_weakness', 'acidity', 'loss_of_balance', 'sweating', 'red_sore_around_nose', 'blurred_and_distorted_vision', 'pain_behind_the_eyes', 'small_dents_in_nails', 'visual_disturbances', 'blackheads', 'continuous_sneezing', 'mood_swings', 'history_of_alcohol_consumption', 'brittle_nails', 'throat_irritation', 'dark_urine', 'neck_pain', 'pain_during_bowel_movements', 'chills', 'fluid_overload', 'acute_liver_failure', 'sunken_eyes', 'increased_appetite', 'vomiting', 'restlessness', 'spinning_movements', 'loss_of_smell', 'weight_gain', 'obesity', 'stiff_neck', 'passage_of_gases', 'back_pain', 'dischromic _patches', 'bruising', 'toxic_look_(typhos)', 'swelled_lymph_nodes', 'spotting_ urination', 'irritation_in_anus', 'pus_filled_pimples', 'irritability', 'prominent_veins_on_calf', 'nodal_skin_eruptions', 'bloody_stool', 'high_fever', 'blister', 'joint_pain', 'abdominal_pain', 'movement_stiffness', 'yellowing_of_eyes', 'scurring', 'irregular_sugar_level', 'inflammatory_nails', 'fatigue', 'cough', 'patches_in_throat', 'swollen_extremeties', 'stomach_bleeding', 'muscle_wasting', 'red_spots_over_body', 'swollen_legs', 'stomach_pain', 'indigestion', 'anxiety', 'lethargy', 'dehydration', 'slurred_speech', 'burning_micturition', 'malaise', 'bladder_discomfort', 'internal_itching', 'redness_of_eyes', 'foul_smell_of urine', 'belly_pain', 'palpitations', 'itching', 'excessive_hunger', 'weakness_in_limbs', 'pain_in_anal_region', 'hip_joint_pain', 'skin_rash', 'painful_walking', 'silver_like_dusting', 'muscle_pain', 'dizziness', 'enlarged_thyroid', 'blood_in_sputum', 'puffy_face_and_eyes', 'abnormal_menstruation', 'yellow_crust_ooze', 'extra_marital_contacts', 'drying_and_tingling_lips', 'chest_pain', 'mild_fever', 'shivering', 'diarrhoea', 'receiving_blood_transfusion', 'rusty_sputum', 'yellowish_skin', 'swollen_blood_vessels', 'cramps', 'weakness_of_one_body_side', 'polyuria', 'congestion', 'constipation', 'distention_of_abdomen', 'swelling_joints', 'family_history', 'fast_heart_rate', 'coma', 'continuous_feel_of_urine', 'knee_pain', 'watering_from_eyes', 'sinus_pressure', 'cold_hands_and_feets', 'swelling_of_stomach', 'phlegm', 'runny_nose', 'loss_of_appetite', 'ulcers_on_tongue', 'altered_sensorium', 'receiving_unsterile_injections', 'depression', 'headache', 'skin_peeling', 'mucoid_sputum', 'nausea', 'weight_loss', 'breathlessness', 'unsteadiness', 'yellow_urine', 'lack_of_concentration']
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
        user_input = " ".join(selected_symptoms)
        predicted_disease = model.predict([user_input])[0]

        st.session_state.predicted_disease = predicted_disease
        st.session_state.selected_symptoms = selected_symptoms
        st.session_state.patient_info = {
            'Name': patient_name,
            'Age': patient_age,
            'Gender': patient_gender,
            'Location': patient_location
        }


        st.markdown(f"""
     
        ### 🔬 **Predicted Disease:**
        🦠 **{predicted_disease}**
        """)
                # Load recorded data
        try:
            df = pd.read_csv('dataset/sample_user_data.csv')
            disease_counts = count_cases(df)
            detected_outbreaks = detect_outbreak_per_day(disease_counts)
            forecasted_outbreaks = predict_future_outbreaks(disease_counts, days=7)

            # Tag historical vs forecasted
            split_date = detected_outbreaks['Date'].max()
            forecasted_outbreaks['source'] = forecasted_outbreaks['Date'].apply(
                lambda d: 'historical' if pd.to_datetime(d) <= pd.to_datetime(split_date) else 'forecasted'
            )

            # Filter data for the predicted disease
            disease_data = forecasted_outbreaks[forecasted_outbreaks['Disease'] == predicted_disease]
            if not disease_data.empty and disease_data['Outbreak'].any():
                st.markdown("### 🚨 **Outbreak Detected for this Disease!**")

                fig, ax = plt.subplots(figsize=(10, 5))
                hist = disease_data[disease_data['source'] == 'historical']
                forecast = disease_data[disease_data['source'] == 'forecasted']

                ax.plot(hist['Date'], hist['Cases'], label='Historical', marker='o', color='blue')
                if not forecast.empty:
                    ax.plot(forecast['Date'], forecast['Cases'], label='Forecasted', marker='o', color='orange')

                outbreak_dates = disease_data[disease_data['Outbreak'] == 1]['Date']
                outbreak_cases = disease_data[disease_data['Outbreak'] == 1]['Cases']
                ax.scatter(outbreak_dates, outbreak_cases, color='red', label='Outbreak', zorder=5)

                ax.set_title(f'📈 Disease Cases and Outbreaks: {predicted_disease}')
                ax.set_xlabel('Date')
                fig.autofmt_xdate()
                ax.set_ylabel('Cases')
                ax.legend()
                st.pyplot(fig)

                # Show user details for outbreak dates
                outbreak_rows = disease_data[disease_data['Outbreak'] == 1]
                latest_date = disease_data['Date'].max()
                if latest_date not in outbreak_rows['Date'].values:
                    latest_row = disease_data[disease_data['Date'] == latest_date]
                    outbreak_rows = pd.concat([outbreak_rows, latest_row])

                if not outbreak_rows.empty:
                    st.markdown(f"**User details for {predicted_disease} Outbreaks:**")
                    for _, row in outbreak_rows.iterrows():
                        outbreak_date = row['Date']
                        matching_users = df[
                            (pd.to_datetime(df['Date']) == pd.to_datetime(outbreak_date)) &
                            (df['Disease'] == predicted_disease)
                        ][['Name', 'Age', 'Gender', 'Location']]
                        if not matching_users.empty:
                            num_cases = len(matching_users)
                            with st.expander(f"Outbreak on {outbreak_date} ({num_cases} case{'s' if num_cases != 1 else ''})"):
                                st.dataframe(matching_users.reset_index(drop=True))

        except Exception as e:
            st.error(f"Error while checking outbreak status: {e}")


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

            """ 
            #ilipat sa disease prediction
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

            """


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


def display():
    st.subheader("Patient Symptom and Diagnosis Log")
    user_data = pd.read_csv('dataset/sample_user_data.csv')
    user_data_limited = user_data.iloc[:, :]
    st.dataframe(user_data_limited, use_container_width=True, height=780)  # scrollable view
    
def main():
    tab1, tab2, tab3 = st.tabs(["🩺 Disease Prediction", "📊 Outbreak Forecasting", '📝 Patient Symptom and Diagnosis Log'])
    with tab1:
        symptom_checker()
    with tab2:
        outbreak_forecasting()
    with tab3:
        display()


if __name__ == "__main__":
    main()
