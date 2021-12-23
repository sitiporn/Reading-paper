
import yaml 
import json
import re




# read config using yaml file 

def read_file_config(path:str):
    
    with open(path) as file:

        yaml_data = yaml.safe_load(file)
        
        jAll = json.dumps(yaml_data)

        loader = yaml.SafeLoader

        loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

        yaml_data = yaml.load(jAll, Loader=loader) 

    return  yaml_data 





