# Metal Concentration Analyzer

A Streamlit app that identifies Al or Mn concentration in a test-tube solution
by extracting the RGB colour with Gemini Vision AI and matching it to lab data.

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Get a FREE Gemini API key:
   https://aistudio.google.com/

3. Replace the sample CSV files with YOUR lab data:
   - al_data.csv  -> columns: R, G, B, Al_concentration_ppm
   - mn_data.csv  -> columns: R, G, B, Mn_concentration_ppm

4. Run the app:
   streamlit run app.py

## CSV Format

Your CSVs must have exactly these columns:

al_data.csv:
  R, G, B, Al_concentration_ppm

mn_data.csv:
  R, G, B, Mn_concentration_ppm

Each row = one calibration colour and its known concentration.

## How It Works

1. User selects Al or Mn test
2. User uploads a photo of the test tube
3. Gemini 1.5 Flash extracts the dominant RGB of the solution
4. App computes Euclidean distance to every row in the CSV
5. Closest match returned with its concentration value
