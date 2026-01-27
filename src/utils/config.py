import yaml
import logging.config
import datetime



def getConfigYAML(conf_file:str) -> any:
    with open(conf_file, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config

def getLogger(name:str, config_file:str="config/logger.yaml") -> logging.Logger:
    
    config = getConfigYAML(config_file)
    if name == "staging" or name == "production":
        config['handlers']['file']['filename'] = "logs/"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".log"
    else:
        config['handlers']['file']['filename'] = "logs/dev.log"
    logging.config.dictConfig(config)

    logger = logging.getLogger(name)
    return logger