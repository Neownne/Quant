import os
from jinja2 import Environment, FileSystemLoader
from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# cache_size=0 disables the Jinja2 template cache to work around
# a Python 3.14 compatibility issue with LRUCache
env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    cache_size=0,
)

templates = Jinja2Templates(env=env)
