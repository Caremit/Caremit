# Caremit
SDG Hackathon

# Installation

In order to install this package in editable mode, do the following:
1. open terminal, move to your favourite project folder
2. run `git clone https://github.com/Caremit/Caremit` (or `git clone git@github.com:Caremit/Caremit.git` for ssh)
3. run `python -m pip install -e Caremit` (choose your favourite python executable, and if you don't need editable mode, drop the `-e` flag)

enjoy :)

# Model Deployment
To deploy a trained model to a public azure endpoint:
1. be in the lucky position that someone set up a cozy azureml environment for you
2. Um, are you still there?
3. find out where the model file is located e.g. /path/to/model/saved_model.pb
4. run the deploy_model.py script `python deploy_model.py /path/to/model  # do not reference the file, only the folder`

# Inspiration

The baseline for our model was presented in [this blogpost](https://medium.com/@CVxTz/heartbeat-classification-detecting-abnormal-heartbeats-and-heart-diseases-from-ecgs-913449c2665) with [code released on MIT on github](https://github.com/CVxTz/ECG_Heartbeat_Classification).

The data preprocessing follows the specification in [this paper](https://arxiv.org/pdf/1805.00794.pdf).


# Data Source

From kaggle we can directly download processed data for both ptbdb and :
https://www.kaggle.com/shayanfazeli/heartbeat?select=mitbih_train.csv

While raw data can be found here:
- mitdb data: download [the zip file](https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip) from this webpage https://www.physionet.org/content/mitdb/1.0.0/ 
- ptbdb data: run `wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/`

More information about the data and how to access it can be found
- https://physionet.org/ (new website)
- https://archive.physionet.org/ (old website)
- https://github.com/MIT-LCP/wfdb-python (client for python)


# Plotting ECG data with wfdb

A nice jupyternotebook demo about hwo to use and plot wfdb can be found here https://github.com/MIT-LCP/wfdb-python/blob/master/demo.ipynb

From there. we crystallized that the following seems the easiest way to get access to signals and annotations for both further processing and plotting.

Example from the notebook where we started
```python
# Demo 5 - Read a WFDB record and annotation. Plot all channels, and the annotation on top of channel 0.
import wfdb
record = wfdb.rdrecord('sample-data/100', sampto = 15000)
annotation = wfdb.rdann('sample-data/100', 'atr', sampto = 15000)

wfdb.plot_wfdb(record=record, annotation=annotation,
               title='Record 100 from MIT-BIH Arrhythmia Database',
               time_units='seconds')
```

it turns out that this [`plot_wfdb`](https://github.com/MIT-LCP/wfdb-python/blob/8269d411513d41370931d62abaf8bbc56053cc2a/wfdb/plot/plot.py#L632-L642) function uses an internal function to extract just everything needed, and then passes those to the more basic `wfdb.plot.plot.plot_items`.

```python
(signal, ann_samp, ann_sym, fs, ylabel, record_name, sig_units) = wfdb.plot.plot.get_wfdb_plot_items(record, annotation, True)
```
We even have detailed documentation about everything by looking into `wfdb.plot.plot.plot_items`

`ylabel` contains the names of the signals, `record_name` is used as title and just contains the filename, all others are best described in the documentation:
```
    signal : 1d or 2d numpy array, optional
        The uniformly sampled signal to be plotted. If signal.ndim is 1, it is
        assumed to be a one channel signal. If it is 2, axes 0 and 1, must
        represent time and channel number respectively.
    ann_samp: list, optional
        A list of annotation locations to plot, with each list item
        corresponding to a different channel. List items may be:
        - 1d numpy array, with values representing sample indices. Empty
          arrays are skipped.
        - list, with values representing sample indices. Empty lists
          are skipped.
        - None. For channels in which nothing is to be plotted.
        If `signal` is defined, the annotation locations will be overlaid on
        the signals, with the list index corresponding to the signal channel.
        The length of `annotation` does not have to match the number of
        channels of `signal`.
    ann_sym: list, optional
        A list of annotation symbols to plot, with each list item
        corresponding to a different channel. List items should be lists of
        strings. The symbols are plotted over the corresponding `ann_samp`
        index locations.
    fs : int, float, optional
        The sampling frequency of the signals and/or annotations. Used to
        calculate time intervals if `time_units` is not 'samples'. Also
        required for plotting ECG grids.
    ylabel : list, optional
        A list of strings specifying the final y labels. If this option is
        present, `sig_name` and `sig_units` will not be used for labels.
    sig_units : list, optional
        A list of strings specifying the units of each signal channel. Used
        with `sig_name` to form y labels, if `ylabel` is not set. This
        parameter is required for plotting ECG grids.
```

# App Development with Flask

Our flask app follows a tutorial given by Azure https://docs.microsoft.com/en-us/azure/active-directory-b2c/tutorial-web-app-python?tabs=app-reg-ga in order to connect with Azure AD B2C. The style is adapted from the MIT-licensed open template ["new-age"](https://startbootstrap.com/theme/new-age).

# Apps Development with Flutter

login demos which somehow worked and looked okay:
- [blogpost](https://levelup.gitconnected.com/login-page-ui-in-flutter-65210e7a6c90), [github](https://github.com/yogitakumar/logindemo)
- [blogpost](https://codesource.io/build-a-simple-login-page-and-dashboard-with-flutter/), [github](https://github.com/ariefsn/simple_login_page)

azure authentication
- [aad_oauth, official flutter package](https://pub.dev/packages/aad_oauth)
- [msal_flutter, official flutter package](https://pub.dev/packages/msal_flutter)
- [blogpost](https://www.detroitdave.dev/2020/04/simple-azure-b2c-flutter.html), [github](https://github.com/dwhiteddsoft/flutter-azure-b2c-appauth)
- [another blogpost](https://medium.com/flutter-community/flutter-azure-authentication-with-ad-b2c-8b76c81dd48e), [github](https://github.com/WJayesh/prod_app)


Setup azure active directory B2C is necessarily a manual process (limitation of azure). You need to follow this tutorial
- https://docs.microsoft.com/en-us/azure/active-directory-b2c/tutorial-create-tenant

