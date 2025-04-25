# ftir_app
The ftir_app allows easy evaluation of a compound's concentration from its FTIR spectrum. The app contains following functions: Spectral data upload (CSV files), ALS baseline smoothing (Eilers and Boelens), peak area calculation, calibrant line generation, concentration evaluation of analyte. 

Works with Python 3.10.6!


### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r ftir_requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run ftir_streamlit.py
   ```

