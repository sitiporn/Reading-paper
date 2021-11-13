import yaml

#read yaml file

with open('config/config.yaml') as file:
    yaml_data = yaml.safe_load(file)


print(yaml_data["model_params"]["model"])
print(yaml_data["model_params"]["classify"])
print(yaml_data["training_params"]["N"])
print(yaml_data["training_params"]["batch_size"])

