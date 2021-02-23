import uuid
import requests
from flask import Flask, render_template, flash, session, request, redirect, url_for
from flask_session import Session  # https://pythonhosted.org/Flask-Session
import msal
import app_config
import json
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
import tempfile
from io import StringIO, BytesIO
import wfdb
from pathlib import Path
from caremit.preprocessing.kachuee_2018 import preprocess_kachuee_2018
import mpld3
# got hint from https://stackoverflow.com/questions/53684971/assertion-failed-flask-server-stops-after-script-is-run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')


BEAT_MAP = {
    0: 'Normal beat (N)',
    1: 'Premature or ectopic supraventricular beat (S)',
    2: 'Premature ventricular contraction (V)',
    3: 'Fusion of ventricular and normal beat (F)',
    4: 'Unclassifiable beat (Q)'
}

app = Flask(__name__)
app.config.from_object(app_config)
Session(app)

# This section is needed for url_for("foo", _external=True) to automatically
# generate http scheme when this sample is running on localhost,
# and to generate https scheme when it is deployed behind reversed proxy.
# See also https://flask.palletsprojects.com/en/1.0.x/deploying/wsgi-standalone/#proxy-setups
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

@app.route("/")
def index():
    if not session.get("user"):
        return redirect(url_for("landing"))
    return render_template('dashboard.html', user=session["user"], version=msal.__version__)

@app.route("/landing")
def landing():
    # Technically we could use empty list [] as scopes to do just sign in,
    # here we choose to also collect end user consent upfront
    session["flow"] = _build_auth_code_flow(scopes=app_config.SCOPE)
    return render_template("landing.html", auth_url=session["flow"]["auth_uri"], version=msal.__version__)

@app.route(app_config.REDIRECT_PATH)  # Its absolute URL must match your app's redirect_uri set in AAD
def authorized():
    try:
        cache = _load_cache()
        result = _build_msal_app(cache=cache).acquire_token_by_auth_code_flow(
            session.get("flow", {}), request.args)
        if "error" in result:
            return render_template("auth_error.html", result=result)
        session["user"] = result.get("id_token_claims")
        _save_cache(cache)
    except ValueError:  # Usually caused by CSRF
        pass  # Simply ignore them
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()  # Wipe out user and its token cache from session
    return redirect(  # Also logout from your tenant's web session
        app_config.AUTHORITY + "/oauth2/v2.0/logout" +
        "?post_logout_redirect_uri=" + url_for("index", _external=True))

@app.route("/machinelearning", methods=['POST'])
def machinelearning():
    if not session.get("user"):
        return redirect(url_for("landing"))

    # check if the post request has the file part
    if 'file' not in request.files:
        msg = 'No file part'
        flash(msg); print(msg)
        return redirect(url_for("index"))
    files = request.files.getlist("file")
    # if user does not select file, browser also
    # submit an empty part without filename

    if (len(files) != 2 
        or {Path(file.filename).suffix for file in files} != {'.dat', '.hea'}
        or len({Path(file.filename).stem for file in files}) != 1):
        msg = "Nead exactly one XXX.dat file and one XXX.hea file where XXX needs to be the same."
        flash(msg); print(msg)
        return redirect(url_for("index"))

    # taken from https://github.com/CVxTz/ECG_Heartbeat_Classification/blob/master/code/baseline_mitbih.py#L11-L19
    with tempfile.TemporaryDirectory() as tmpdirname:
        for file in files:
            filename = secure_filename(Path(file.filename).name)
            filepath = os.path.join(tmpdirname, filename)
            file.save(filepath)

        # file path without extension is guaranteed to be the same for both files
        prefix = os.path.splitext(filepath)[0]
        record = wfdb.rdrecord(prefix)

    single_signal = record.p_signal[:, 0]
    fs = record.fs
    segments, start_end = preprocess_kachuee_2018(single_signal, fs, heartbeat_position="left")
    data = segments[..., np.newaxis]

    token = _get_token_from_cache(app_config.SCOPE)
    if not token:
        # not authorized, hence going back to login page
        return redirect(url_for("landing"))

    # speak to endpoint
    confidence_levels = requests.post(
        app_config.ENDPOINT,
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token['id_token']},
        data=json.dumps(data.tolist())
    ).json()
    html = create_plot_highest_conf(data, confidence_levels)
    return render_template('display.html', user=session["user"], result=html, version=msal.__version__)

def _load_cache():
    cache = msal.SerializableTokenCache()
    if session.get("token_cache"):
        cache.deserialize(session["token_cache"])
    return cache

def _save_cache(cache):
    if cache.has_state_changed:
        session["token_cache"] = cache.serialize()

def _build_msal_app(cache=None, authority=None):
    return msal.ConfidentialClientApplication(
        app_config.CLIENT_ID, authority=authority or app_config.AUTHORITY, token_cache=cache)  # , client_credential=app_config.CLIENT_SECRET

def _build_auth_code_flow(authority=None, scopes=None):
    return _build_msal_app(authority=authority).initiate_auth_code_flow(
        scopes or [],
        redirect_uri=url_for("authorized", _external=True))

def _get_token_from_cache(scope=None):
    cache = _load_cache()  # This web app maintains one cache per session
    cca = _build_msal_app(cache=cache)
    accounts = cca.get_accounts()
    if accounts:  # So all account(s) belong to the current signed-in user
        result = cca.acquire_token_silent(scope, account=accounts[0])
        _save_cache(cache)
        return result


def create_plot_highest_conf(signal_data: np.ndarray, confidence_levels):
    """Plots the signal with the highest confidence per encountered
    category, if the prediction was correct.
    Raises if highest prediction per category was incorrect."""

    confidence_levels_df = pd.DataFrame(confidence_levels)
    confidence_levels_df['predicted_label'] = confidence_levels_df.idxmax(axis=1)
    confidence_levels_df['confidence'] = confidence_levels_df.loc[:, list(range(5))].max(axis=1)
    
    def keep_max_confidence_only(df_group):
        max_idx = df_group['confidence'].idxmax()
        subdf = df_group.loc[max_idx:max_idx, :]
        subdf['max_idx'] = max_idx
        subdf['count'] = len(df_group)
        return subdf
    
    overview = confidence_levels_df \
        .loc[:, ['predicted_label', 'confidence']] \
        .groupby('predicted_label') \
        .apply(keep_max_confidence_only)
    
    fig, axs = plt.subplots(len(overview), 1, sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for ax, row in zip(axs, overview.itertuples()):
        signal = signal_data[row.max_idx]
        ax.plot(signal, color="cyan")
        ax.set_title(f"Representative signal for category '{BEAT_MAP[row.predicted_label]}'")
        ax.set_xlabel("time in ms")
        ax.set_ylabel("relative signal strength")
        [t.set_color('white') for t in ax.xaxis.get_ticklines()]
        [t.set_color('white') for t in ax.xaxis.get_ticklabels()]
        ax.grid(color=(0.01, 0.01, 0.01), linestyle=":", linewidth=0.5)

    fig.tight_layout()

    html_overview = overview.to_html()
    html_plot = mpld3.fig_to_html(fig)
    return f"""
        <div class="row justify-content-center align-self-center">
            {html_overview}
        </div>
        <div class="row overflow-auto">
            {html_plot}
        </div>
    """
        

app.jinja_env.globals.update(_build_auth_code_flow=_build_auth_code_flow)  # Used in template

if __name__ == "__main__":
    app.run()

