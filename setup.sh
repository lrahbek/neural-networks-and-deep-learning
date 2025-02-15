python -m venv env
source ./env/bin/activate

python -m pip install ipykernel
python -m ipykernel install --user --name=env

pip install --upgrade pip
pip install -r req2.txt
deactivate
