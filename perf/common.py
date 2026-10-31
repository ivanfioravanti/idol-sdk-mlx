import glob, json, os
WEIGHTS = sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--speridlabs--idol/snapshots/*')))[-1]
CONFIG = json.load(open(os.path.join(WEIGHTS, 'config.json')))
IMAGES = {'avocado': 'demo/examples/food-avocado.webp', 'fox': 'demo/examples/fig-fox.webp'}
