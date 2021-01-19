# artemis-bidirectional-compression
Artemis: fast convergence guarantees for bidirectional compression in Federated Learning

This code has been written by Constantin Philippenko, and is a jointly work with Aymeric Dieuleveut 
at Ecole Polytechnique.

## Experimentation

All the illustrations have been generated using the script `experiments_runner.py`. 
There is 8 parameters:

1. `nb_devices: int`.
2. `stochastic: bool`.
3. `dataset: str`. Possible values are : ['quantum', 'superconduct', 'synth_linear_noised', 'synth_linear_nonoised']. 
   The first two dataset are real, the three last are synthetics : two using the RMSE model, and one the logistic model. 
4. `iid: str`. Possible values are ['iid', 'non-iid']. For synthetic dataset this option has no impact as linear models are 
   using iid data, and logistic models are using non-iid data.
5. `algos: str`. Possible values are ["uni-vs-bi", "with-without-ef", "compress-model"]. This corresponds to the different 
   kind of algorithms we want to run. The first option analyzes the impact of bidirectionnel compression and of memory. 
   The second option compares the impact of the error-feedback process. The last one compares algorithms compressing 
   (or not) the model for the way back.
6. `use_averaging: bool`. Set to true if you want to also have result for the Polyak-Ruppert averaging.
7. `scenario: str`. Possible values are [None, "compression", "step"]. The first option will run with the default step 
   size and the default compression. The second option will run experiments for various level of compression.
   Last one will run experiments for different step size.
8. `plot_only: bool`. Set to true if you do not want to re-run experiments and only re-generate figures.

We provide all the notebooks used to investigate datasets in the folder `notebook/`. 
To regenerate all the figures, just restart the notebook.  
However we warn that they need to run for about 2hours to 12hours. 
Yet, it is possible (using pickle) to load already generated data and to explore them. To do it: 

1. Load pickle methods: `from src.utils.Utilities import pickle_loader`
2. Load data: `pickle_loader(<the-file-you-want-to-load>)`

## Code structuration

In `src/` is gathered all the code. 
In `notebook/` is provided all notebook used for our experimentation.
In `test/` are written some test of our code. Presently, the coverage of our code is very little.

The aim of this github repository is to implement Artemis. It has been designed in a way which allows high 
flexibility. Artemis can be ran with various features, namely:
    
1. Gradients or models compression
2. Bidirectional or unidirectional compression
3. Stochastic or full batch descent
4. With or without memory
5. With one or two memories (one for each way)
6. With or without momentum
7. With or without Polyak-Ruppert averaging
8. With or without devices sampling
9. ...

Other features are planned to be coded to enhance the code and the possibilities. In this perspective, the code 
has been designed to easily implement other kind of algorithm. This is why, it presents an important
part of abstract code. 
The code is split into three packages:

1. `machinery/`: It contains all the main classes designed to run the gradient descent. We are at the core of the 
learning factory, in a sense, it corresponds to its machinery. This package includes a class to:

    1. run the proper descent (AGradientDescent)
    2. update the model (AGradientDescentUpdate)
    3. update the local model when it is relevant i.e in federated lerning settings (LocalUpdate)
    4. set all the parameters of the run (Parameters) 
    5. modelize each workers of the network (Worker)

2. `model/`: In this package, we gather all class defining a model for a particular object of the learning problem.
 We defined a model for the cost function, the regularization and the quantization.

3. `utilities/`: In this package are located all classes required to run experiments but which are not related to the 
proper algorithm. The subpackage `runner/`. aims to provide functions to easily run multiple gradient descent and 
agregate their results. 

## How to run the code on remote server from a linux laptop

1. Zip the code and the notebook folders: `tar -zcvf mycode.tar.gz notebook src`
2. Send on remote server: `scp mycode.tar.gz user@server:~`
3. Unzip: `tar -xzvf mycode.tar.gz`
4. Run notebook in commande line.

## How to code with notebook

### Importing modules in Notebook
Use: `import sys; sys.path.insert(0, '..')`. Then just import code as usually `from src.mymodule import something`

### Fancy gadgets
To display cell execution time, use following widget: `%%time`
To display progress bar, use `tqdm` package : 
`import from tqdm import tqdm; for i in tqdm(range(10000)): ...`

## Test

### Running all unit test :
`python -m unittest discover`

### Code covering 

Use https://coverage.readthedocs.io/en/coverage-5.1/.
With cobertura : `coverage run -m unittest discover`
Then to see report : `coverage report`

### Running Doctests :
Run `python -m doctest *.py -v`

