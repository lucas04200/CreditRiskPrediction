import streamlit as st
import requests

# Header
st.set_page_config(page_title="Prédiction Client", page_icon=":guardsman:", layout="centered")
st.title("Formulaire de Prédiction")

# Form
with st.form("input_form"):
    input1 = st.text_input("Entrez l'ID Client :")
    input2 = st.text_input("Entrez le montant :")
    input3 = st.text_input("Entrez le taux d'intérêt (%) :")
    
    submitted = st.form_submit_button("Soumettre", use_container_width=True)

# Body
if submitted:
    try:
        # Convertir les entrées en valeurs numériques
        amount = float(input2)
        interest = float(input3)

        # Préparer les données pour l'API
        data = {
            "ID": input1,
            "Amount": amount,
            "Interest": interest
        }

        # Faire la requête à l'API
        url = "http://fastapi_app:8000/predict"
        response = requests.post(url, json=data)

        if response.status_code == 200:
            response_data = response.json()
            st.json(response_data)

            if "Prediction" in response_data:
                st.success(f"Client {response_data['ID']} : {response_data['Prediction']}")
            else:
                st.error("Erreur dans la prédiction")
        else:
            st.error("Erreur de communication avec l'API")

    except ValueError:
        st.error("Veuillez entrer des valeurs valides pour le montant et le taux d'intérêt.")
else: 
    st.write("Veuillez remplir le formulaire et cliquer sur le bouton Submit.")

# Footer
st.markdown("""
---
Si vous avez des questions, n'hésitez pas. :D
""", unsafe_allow_html=True)