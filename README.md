# Primus-Turbo


## Docker Image
```
rocm/megatron-lm:v25.5_py310
```

## Install
```
git clone https://github.com/AMD-AIG-AIMA/Primus-Turbo.git
git submodule update --init --recursive
```

### Develop
```
pip3 install -r requirements.txt
python3 setup.py develop
pytest -s ./tests/
```

### Deploy
```
python3 setup.py bdist_wheel
```
