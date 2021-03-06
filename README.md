# fnet-web: The web front-end of AI-reporter

## Install dependencies

First, you need to install the training backend. We currently use [BNNBench](https://github.com/xuanqing94/BNNBench), so we install it first with

```bash
pip install git+https://github.com/xuanqing94/BNNBench.git@v0.1#egg=BNNBench
```

Then, install other dependencies
```bash
pip install -r requirements.txt
```

## Prepare folders for data and checkpoints

You need to manually create following folders to store datasets, model checkpoints and results. The program assumes all folders exist, otherwise it will exit with error.

```bash
# in fnet-web/ folder
mkdir third_party_data checkpoints results
```

## Start server

You can simply run with

```bash
python app.py
```

Then open browser and visit (https://localhost:8991)

## Serious deployment

The section `Start server` above is only good for casual use. If this service is open to public in the future, we need to run on a production grade server. This [link](https://flask.palletsprojects.com/en/2.0.x/tutorial/deploy/#run-with-a-production-server) tells you how.
