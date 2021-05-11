## From Notebook to package 🎁

It is time to move away from Jupyter Notebook, and start writing reusable code with python packages, modules and classes.

In the exercise, nothing new, simply copy pasting functions we already implemented in the previous challenge and organize them inside different modules.

### Package structure 🗺

We have created for you the following structure:

```bash
├── TaxiFareModel
│   ├── data.py       # functions to get and clean data
│   ├── encoders.py   # your custom encoders and transformers for our Pipeline
│   ├── trainer.py    # utility functions
│   └── utils.py      # containing main class that will run our Pipeline
```

### Setup your package ⚙️

You are going to create a package from your pipeline. To achieve this, we provide you a minimal template generator package [`packgenlite`](https://github.com/krokrob/packgenlite).

- Install the `packgenlite` package from Github

```bash
pip install git+https://github.com/krokrob/packgenlite
```

- Create a new project `TaxiFareModel` in your working directory

```bash
cd ~/code/<user.github_nickname>
packgenlite TaxiFareModel
```

- Copy the code we provide into your project

<details>
  <summary markdown='span'><strong>💡 Hint</strong></summary>

<br>

```bash
cp -r ~/code/<user.github_nickname>/data-challenges/07-Data-Engineering/02-ML-Iteration/03-Notebook-to-package/*.py ~/code/<user.github_nickname>/TaxiFareModel/TaxiFareModel
```

</details>

- Make sure your package has the following structure
```bash
.
├── MANIFEST.in
├── Makefile
├── README.md
├── TaxiFareModel
│   ├── __init__.py
│   ├── data
│   ├── data.py
│   ├── encoders.py
│   ├── trainer.py
│   └── utils.py
├── notebooks
├── raw_data
├── requirements.txt
├── .gitignore
├── scripts
│   └── TaxiFareModel-run
├── setup.py
└── tests
    └── __init__.py
```

👍 Your package is ready to be implemented!

### Implement your package 🛠

#### Download the datasets locally

- Download the datasets `train.csv` and `test.csv` from [Kaggle (`Data` tab)](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data) if you have not done it yet
- Move them under the `raw_data` folder
- Download 2 subsets of the full `train.csv` dataset:
  - [train_1k.csv](https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-ny/train_1k.csv) with 1_000 rows
  - [train_10k.csv](https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-ny/train_10k.csv) with 10_000 rows
- Move them to the `raw_data` folder
- Make sure your package has the following architecture:

```bash
.
├── MANIFEST.in
├── Makefile
├── README.md
├── TaxiFareModel
│   ├── __init__.py
│   ├── data
│   ├── data.py
│   ├── encoders.py
│   ├── trainer.py
│   └── utils.py
├── notebooks
├── raw_data
│   ├── test.csv
│   ├── train.csv
│   ├── train_10k.csv
│   └── train_1k.csv
├── requirements.txt
├── .gitignore
├── scripts
│   └── TaxiFareModel-run
├── setup.py
└── tests
    └── __init__.py
```

#### `data.py`

Inspect the functions `get_data` and `clean_data` that are already given to you.

_NB: We provide you with the same functions so that we all get and clean data the same way._

#### `utils.py`

`utils.py` is where you can store :
- the `haversine_distance` method
- the `compute_rmse` method

#### `encoders.py`

Let's store the custom encoders and transformers you have for distance and time features in `encoders.py`.

**Reminder**

Let us be clear about the use of `encoders.py` here:
- It contains all the custom pipeline preprocessing blocks not provided by sklearn
- These preprocessing blocks are the `DistanceTransformer` and the `TimeFeaturesEncoder`

#### `trainer.py`

Implement the main class here.

The `Trainer` class is the main class. It should have:
- an `__init__` method called when the class is instanciated
- a `set_pipeline` method that builds the pipeline
- a `run` method that trains the pipeline
- an `evaluate` method evaluating the model

Make sure you are confident with the following notions:
- attributes and methods of a class
- `**kwargs` argument of a function and how to use it, (help [HERE](https://www.programiz.com/python-programming/args-and-kwargs) if unclear)

```python
class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

    def run(self):
        """set and train the pipeline"""

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
```

#### Test your package!

Once you have everything setup, test that it works by running:

```bash
python -m TaxiFareModel.trainer
```

Or

```bash
python -i TaxiFareModel/trainer.py
```

#### Hints for debugging 🐛

- Do not hesitate to breakdown your code into smaller calls for debugging
- Use [if \_\_name__ == '\_\_main__'](https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/) at the end of each `.py` file to debug
- For instance to debug data.py, add at the end:

```python
# TaxiFareModel/data.py

if __name__ == '__main__':
    df = get_data()
```

Then in an `ipython` session from your terminal, you can run:

```bash
In [1]: %run TaxiFareModel/data.py

In [2]: df.shape
```

### Install your package

When your package is all set, you should install it locally so that it can be imported anywhere. From your package folder, run:

```bash
pip install -e .
```

Then you can open the `taxi_fare_model_package_testing.ipynb` and run the cells.

### Push your package on GitHub 🐙😸

1. Create a repository on GitHub
2. Add a remote between your package and the GitHub repository
3. Push your code on the GitHub repository

👍 Good job!
