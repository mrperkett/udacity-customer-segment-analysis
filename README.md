# Introduction
This is my submission for the unsupervised learning project as part of the Udacity nanodegree [Introduction to Machine Learning with TensorFlow](https://www.udacity.com/course/intro-to-machine-learning-with-tensorflow-nanodegree--nd230). The point of this project is to demonstrate understanding of unsupervised learning concepts and the scikit-learn library through a customer segmentation task.  The high-level goal is to identify different clusters within the general population and then determine to which of these clusters the company's customers below.  By looking at the overrepresentation and underrepresentation of customers in each of the clusters relative to the general population, it is possible to identify groups that are, and are not customers.  This information could then be used to inform the product roadmap or marketing strategy to find new customer groups or reach more likely customers within the same groups.

These are the high level tasks
- Clean data
  - Add NaNs where appropriate
  - drop columns with little data
  - decide on a threshold and drop rows with too much missing data
  - one-hot encode categorical columns
  - engineer new features for any mixed data type columns
- Transform data
  - scale so that all features are on equal footing
  - feature reduction with PCA
- Cluster the general population data using k-means
- Compare customers vs general population using clusters found for the general population

The data for this task comes from Arvato in Germany, and provides a realistic example of data.  Unfortunately, the data must be deleted after project completion, so only the high-level data overview files and analysis can be provided in this repo.



# Setup

## Docker

```
# clone the repo
git clone git@github.com:mrperkett/udacity-project-creating-customer-segments.git
cd udacity-project-creating-customer-segments/

# build the docker image
docker build --rm -t udacity-customer-segments -f Dockerfile .
```

## Local
Start by cloning the repo.

```
git clone git@github.com:mrperkett/udacity-project-creating-customer-segments.git
cd udacity-project-creating-customer-segments/
```

### Set up Jupyter
Set up Jupyter in its own `pyenv` environment.  If you already have Jupyter set up, you can skip this step.

```
# create virtual environment
pyenv install 3.11.7
pyenv virtualenv 3.11.7 jupyter
pyenv activate jupyter

# install jupyter lab
python3 -m pip install --upgrade pip
python3 -m pip install -r jupyter-requirements.txt
pyenv deactivate jupyter
```

### Set up IPython kernel
Set up an IPython kernel with its requirements in its own environment.

```
# create virtual environment
pyenv install 3.11.7
pyenv virtualenv 3.11.7 udacity-customer-segments
pyenv local udacity-customer-segments
python3 -m pip install --upgrade pip

# install requirements
python3 -m pip install -r requirements.txt

# register IPython kernel
python3 -m ipykernel install --user --name udacity-customer-segments
```


# Running
## Docker

```
# run the docker container starting a jupyter lab server
# - mount the current working directory to /work/ in the image
local_work_dir="."
port=8890

docker run -e port=${port} -p ${port}:${port} -it --rm -v "${local_work_dir}":/work udacity-customer-segments
# Connect via your browser, VSCode Dev Containers, or similar
# Select the IPython kernel udacity-customer-segments and then run
```

## Local
```
# start jupyter lab server
pyenv activate jupyter
python3 -m jupyter lab
# Connect via your browser, VSCode, or similar
# Select the IPython kernel udacity-customer-segments and then run
```

# Results

See the Jupyter notebook ([Identify_Customer_Segments.ipynb](Identify_Customer_Segments.ipynb)) for the full analysis.

See the [output](output) folder for all images saved during the analysis.

# Data
Unfortunately, the data must be deleted after project completion, so only the high-level data overview files and analysis can be provided in this repo.