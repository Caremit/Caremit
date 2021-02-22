import uuid
import requests
from flask import Flask, render_template, session, request, redirect, url_for
from flask_session import Session  # https://pythonhosted.org/Flask-Session
import msal
import app_config
import json
import numpy as np


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
        return redirect(url_for("login"))
    return render_template('index.html', user=session["user"], version=msal.__version__)

@app.route("/login")
def login():
    # Technically we could use empty list [] as scopes to do just sign in,
    # here we choose to also collect end user consent upfront
    session["flow"] = _build_auth_code_flow(scopes=app_config.SCOPE)
    return render_template("login.html", auth_url=session["flow"]["auth_uri"], version=msal.__version__)

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

@app.route("/machinelearning")
def machinelearning():
    token = _get_token_from_cache(app_config.SCOPE)
    print(token)
    if not token:
        return redirect(url_for("login"))
    graph_data = requests.post(
        app_config.ENDPOINT,
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token['id_token']},
        data=json.dumps([[[1.0], [0.8402625918388366], [0.0], [0.024070022627711293], [0.010940918698906898], [0.08096279948949814], [0.09628009051084517], [0.10503282397985457], [0.10284464061260223], [0.12035010755062102], [0.10065645724534988], [0.12253829091787337], [0.1072210073471069], [0.13347920775413513], [0.11816192418336867], [0.1356673985719681], [0.1269146651029587], [0.14442013204097748], [0.13785557448863986], [0.1553610563278198], [0.14879649877548218], [0.17505469918251038], [0.16630196571350095], [0.19474835693836207], [0.1859956234693527], [0.2144420146942138], [0.227571114897728], [0.24945294857025144], [0.26258206367492676], [0.2997811734676361], [0.3238511979579925], [0.3676148653030396], [0.3873085379600525], [0.43107220530509943], [0.4551422297954559], [0.4967177212238312], [0.5142232179641724], [0.5448577404022217], [0.52078777551651], [0.4792122542858124], [0.39824944734573364], [0.3304157555103302], [0.2385120391845703], [0.19037199020385745], [0.13347920775413513], [0.12253829091787337], [0.08533916622400285], [0.08096279948949814], [0.061269145458936684], [0.07439824938774109], [0.06345733255147934], [0.07002188265323639], [0.054704595357179635], [0.06783369928598404], [0.056892778724432], [0.07002188265323639], [0.054704595357179635], [0.07002188265323639], [0.06345733255147934], [0.07439824938774109], [0.06345733255147934], [0.07439824938774109], [0.06345733255147934], [0.07221006602048875], [0.056892778724432], [0.07002188265323639], [0.056892778724432], [0.07002188265323639], [0.05032822862267494], [0.06783369928598404], [0.05032822862267494], [0.06345733255147934], [0.041575491428375244], [0.06345733255147934], [0.04595185816287994], [0.06564551591873169], [0.04595185816287994], [0.06345733255147934], [0.05032822862267494], [0.056892778724432], [0.048140045255422585], [0.061269145458936684], [0.05032822862267494], [0.061269145458936684], [0.05032822862267494], [0.056892778724432], [0.05032822862267494], [0.05251641198992729], [0.043763674795627594], [0.059080962091684334], [0.05032822862267494], [0.056892778724432], [0.041575491428375244], [0.054704595357179635], [0.048140045255422585], [0.061269145458936684], [0.05251641198992729], [0.06345733255147934], [0.05251641198992729], [0.06345733255147934], [0.05251641198992729], [0.059080962091684334], [0.048140045255422585], [0.06564551591873169], [0.056892778724432], [0.07658643275499344], [0.07877461612224579], [0.09190371632575987], [0.09190371632575987], [0.08752734959125519], [0.08315098285675049], [0.08315098285675049], [0.05251641198992729], [0.059080962091684334], [0.054704595357179635], [0.07002188265323639], [0.05251641198992729], [0.07002188265323639], [0.07221006602048875], [0.08752734959125519], [0.06345733255147934], [0.024070022627711293], [0.054704595357179635], [0.4288840293884276], [0.969365417957306], [0.9518599510192872], [0.14004376530647275], [0.010940918698906898], [0.0], [0.061269145458936684], [0.09628009051084517], [0.1072210073471069], [0.10065645724534988], [0.11816192418336867], [0.10940919071435927], [0.12472647428512572], [0.11378555744886397], [0.13347920775413513], [0.12035010755062102], [0.1356673985719681], [0.1269146651029587], [0.1422319412231445], [0.13785557448863986], [0.1553610563278198], [0.13347920775413513], [0.1597374230623245], [0.1553610563278198], [0.17286652326583862], [0.17286652326583862], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]])
    ).json()
    return render_template('rdisplay.html', result=graph_data)


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

