import argparse
from azureml.core import Model, Workspace, Environment
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

parser = argparse.ArgumentParser(description='Deploy a model')
parser.add_argument('model', help='path to the saved model that should be deployed (path containing a saved_model.pb file)')
args = parser.parse_args()

ws = Workspace.from_config()

model = Model.register(workspace=ws,
                       model_name='my-tensorflow-model',                # Name of the registered model in your workspace.
                       model_path=args.model,
                       model_framework=Model.Framework.TENSORFLOW,  # Framework used to create the model.
                       model_framework_version='2.4.0',
                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       description='Ridge regression model to predict diabetes progression.',
                       tags={'area': 'diabetes', 'type': 'regression'})

print('Name:', model.name)
print('Version:', model.version)

environment = Environment('my-tf-environment')
environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'tensorflow==2.4.0'
])

service_name = 'my-custom-env-service'

inference_config = InferenceConfig(entry_script='./deployment/entry_script.py', environment=environment)
aci_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=4)

# if this fails there is probably already a service with that name, that should be deleted first
service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)

service.wait_for_deployment(show_output=True)

print("LOGS:")
print(service.get_logs())

