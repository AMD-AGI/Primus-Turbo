# Primus-Turbo


## Docker Image
```
rocm/megatron-lm:latest
```

## Install
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
