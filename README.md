# ftir_app
The ftir_app allows easy evaluation of a compound's concentration from its FTIR spectrum. The app contains following functions: Spectral data upload (CSV files), ALS baseline smoothing (Eilers and Boelens), peak area calculation, calibrant line generation, concentration evaluation of analyte. 

Works with Python 3.10.6! To generate virtual environment in bash do:

curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
source ~/.bashrc

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev git

pyenv install 3.10.6
pyenv virtualenv 3.10.6 venv-3.10.6
pyenv activate venv-3.10.6
pip install -r requirements.txt


### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run ftir_streamlit.py
   ```

