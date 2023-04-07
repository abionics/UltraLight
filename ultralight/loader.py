import logging
import os.path

import requests
from tqdm import tqdm

from ultralight.types import SizeType

MODEL_FOLDER = '.models'
MODEL_URL_TEMPLATE = 'https://github.com/abionics/UltraLight/releases/download/v2.0.0/{}'


class UltraLightLoader:

    def __init__(self, model_folder: str = MODEL_FOLDER, model_url_template: str = MODEL_URL_TEMPLATE):
        self._model_folder = model_folder
        self._model_url_template = model_url_template

    def load(self, size: SizeType, batched: bool) -> str:
        name = f'ultra_light_{size}_batched.onnx' if batched else f'ultra_light_{size}.onnx'
        filename = os.path.join(self._model_folder, name)
        if not os.path.exists(filename):
            url = self._model_url_template.format(name)
            self._download(url, filename)
        return filename

    def _download(self, url: str, filename: str, block_size: int = 8192):
        os.makedirs(self._model_folder, exist_ok=True)
        logging.info(f'Downloading model "{filename}" from {url}...')
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('Content-Length', 0))
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            pbar.set_description('Downloading model')
            with open(filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    file.write(data)
        logging.info(f'Downloaded model "{filename}"')
