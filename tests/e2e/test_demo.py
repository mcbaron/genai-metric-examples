from pathlib import Path

from pydantic import TypeAdapter

from promptx_core.config import Config
from promptx_core.data import API_DEFAULT_DATA_TYPE
from promptx_core.data.dataset import get_prepared_dataset
from promptx_core.data.handler_factory_methods import get_data_handler
from promptx_core.data.utils import read_file
from promptx_core.main import run_optimization
from promptx_core.models.promptx import PromptxOutput
from promptx_core.utils.logger import setup_logger
from promptx_core.utils.tracer import setup_tracer

logger = setup_logger(__name__)
tracer = setup_tracer(__name__)


class TestDemo:
    def test_demo_mipro(self):
        path_to_data = Path(__file__).parents[1]

        config_path = path_to_data / "e2e/demo_config_MIPROv2.yaml"

        config = Config.load(config_path)

        byte_io = read_file(path_to_data / "data/demo_data.json")

        data_handler = get_data_handler(API_DEFAULT_DATA_TYPE)
        data_handler.save_data(byte_io, file_name="saved_data.json")

        data = get_prepared_dataset(
            config=config.prompt.context,
            data_loader=data_handler,
            filename="saved_data.json",
        )

        output = run_optimization(config, data, output_type="detailed")
        assert TypeAdapter(PromptxOutput).validate_python(output)

    def test_demo_bfsrs(self):
        path_to_data = Path(__file__).parents[1]

        config_path = path_to_data / "e2e/demo_config_BFSRandomSearch.yaml"

        config = Config.load(config_path)

        byte_io = read_file(path_to_data / "data/demo_data.json")

        data_handler = get_data_handler(API_DEFAULT_DATA_TYPE)
        data_handler.save_data(byte_io, file_name="saved_data.json")

        data = get_prepared_dataset(
            config=config.prompt.context,
            data_loader=data_handler,
            filename="saved_data.json",
        )

        output = run_optimization(config, data, output_type="model")
        assert TypeAdapter(PromptxOutput).validate_python(output)

    def test_demo_bfswo(self):
        path_to_data = Path(__file__).parents[1]

        config_path = path_to_data / "e2e/demo_config_BFSOptuna.yaml"

        config = Config.load(config_path)

        byte_io = read_file(path_to_data / "data/demo_data.json")

        data_handler = get_data_handler(API_DEFAULT_DATA_TYPE)
        data_handler.save_data(byte_io, file_name="saved_data.json")

        data = get_prepared_dataset(
            config=config.prompt.context,
            data_loader=data_handler,
            filename="saved_data.json",
        )

        output = run_optimization(config, data, output_type="model")
        assert TypeAdapter(PromptxOutput).validate_python(output)
