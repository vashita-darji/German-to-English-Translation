import streamlit as st # type: ignore #type ignore
import requests # type: ignore

# Constants
API_URL = "XYZ"  # Replace with your REST endpoint URL
API_KEY = "abc"  # Replace with your API key

# Function to call the Azure ML endpoint
def get_translation(sentence):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'  # Use Bearer token for authorization
    }

    # Prepare input data for the endpoint (check if this matches your model's input format)
    input_data = ({'sentence': sentence})
    

    try:
        # Make a POST request to the endpoint
        response = requests.post(API_URL, headers=headers, json=input_data)
        
        # Ensure a successful response
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Process the response
        try:
            response_data = response.json()  # Attempt to parse JSON response
        except ValueError:
            return "Error: Failed to parse response from the server. Non-JSON response received."

        # Handle potential errors in the response data
        if isinstance(response_data, dict) and 'translated_sentence' in response_data:
            return response_data['translated_sentence']
        else:
            return f"Error: {response_data.get('error', 'No translation result found.')}"
    
    except requests.exceptions.RequestException as e:
        # Handle connection or HTTP errors
        return f"Request failed: {str(e)}"
    except Exception as e:
        # Handle any other unexpected errors
        return f"An unexpected error occurred: {str(e)}"

# Streamlit UI
st.title("German to English Translation")
st.markdown("Translate German sentences to English using the deployed Azure ML model.")

# Input box for the user to enter text
sentence = st.text_area("Enter a German sentence:")

if st.button("Translate"):
    if sentence.strip():
        with st.spinner("Translating..."):
            result = get_translation(sentence)
        st.success("Translation Complete!")
        st.write("**Translated Sentence:**")
        st.write(result)
    else:
        st.warning("Please enter a sentence to translate.")
