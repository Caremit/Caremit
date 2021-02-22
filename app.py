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

ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# taken from https://stackoverflow.com/questions/47160211/why-doesnt-tempfile-spooledtemporaryfile-implement-readable-writable-seekable
class MySpooledTempfile(tempfile.SpooledTemporaryFile):                                                                                
    @property                                                                                                                          
    def readable(self):                                                                                                                
        return self._file.readable                                                                                                     

    @property                                                                                                                          
    def writable(self):                                                                                                                
        return self._file.writable                                                                                                     

    @property                                                                                                                          
    def seekable(self):                                                                                                                
        return self._file.seekable 

@app.route("/machinelearning", methods=['POST'])
def machinelearning():
    if not session.get("user"):
        return redirect(url_for("landing"))

    print(request.url)
    print(request.files)
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        print("No file part")
        return redirect(url_for("index"))
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename

    if not file or file.filename == '':
        flash('No selected file')
        print("No selected file")
        return redirect(url_for("index"))
    
    if not allowed_file(file.filename):
        flash('Please choose a file with one of the following file-ending: ' + ", ".join(sorted(ALLOWED_EXTENSIONS)))
        print('Please choose a file with one of the following file-ending: ' + ", ".join(sorted(ALLOWED_EXTENSIONS)))
        return redirect(url_for("index"))

    # taken from https://github.com/CVxTz/ECG_Heartbeat_Classification/blob/master/code/baseline_mitbih.py#L11-L19
    df = pd.read_csv(BytesIO(file.read()), header=None)
    data = df.values[..., np.newaxis]

    token = _get_token_from_cache(app_config.SCOPE)
    if not token:
        # not authorized, hence going back to login page
        return redirect(url_for("landing"))

    # speak to endpoint
    graph_data = requests.post(
        app_config.ENDPOINT,
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token['id_token']},
        data=json.dumps(data.tolist())
    ).json()
    return render_template('display.html', user=session["user"], result=graph_data, version=msal.__version__)

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

app.jinja_env.globals.update(_build_auth_code_flow=_build_auth_code_flow)  # Used in template

if __name__ == "__main__":
    app.run()

