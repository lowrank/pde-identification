# PDE Identification

Repository for **PDE Identification** with partial solution data.

- [x] Solution generation
- [ ] Identification

## Experiment Requirements
```bash
apt-get install build-essential
apt-get install libfftw3-dev

pip install -r requirements.txt
python setup.py build_ext -b utilities
```

## Tests
```bash
python runtests.py
```

```bash
python -m unittest discover -s tests -p '*test.py'
```

# Examples

![Example Solutions](docs/example.png)

