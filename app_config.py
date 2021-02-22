import os

b2c_tenant = "caremitb2c"
signupsignin_user_flow = "B2C_1_signupsignin_caremit"
authority_template = "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{user_flow}"

CLIENT_ID = "f615de81-04c9-428a-9055-42b793872fab" # Application (client) ID of app registration

CLIENT_SECRET = "X4fu410Ova1tRauwP--~t6_4g3U9drhfH." # Placeholder - for use ONLY during testing.
# In a production app, we recommend you use a more secure method of storing your secret,
# like Azure Key Vault. Or, use an environment variable as described in Flask's documentation:
# https://flask.palletsprojects.com/en/1.1.x/config/#configuring-from-environment-variables
# CLIENT_SECRET = os.getenv("CLIENT_SECRET")
# if not CLIENT_SECRET:
#     raise ValueError("Need to define CLIENT_SECRET environment variable")

AUTHORITY = authority_template.format(
    tenant=b2c_tenant, user_flow=signupsignin_user_flow)

REDIRECT_PATH = "/getAToken"  # Used for forming an absolute URL to your redirect URI.
                              # The absolute URL must match the redirect URI you set
                              # in the app's registration in the Azure portal.

# This is the API resource endpoint
ENDPOINT = 'http://974c45b8-f1de-43f8-b87a-7f5f2d4c5ff7.westeurope.azurecontainer.io/score' # Application ID URI of app registration in Azure portal

# These are the scopes you've exposed in the web API app registration in the Azure portal
SCOPE = []  # Example with two exposed scopes: ["demo.read", "demo.write"]

SESSION_TYPE = "filesystem"  # Specifies the token cache should be stored in server-side session
