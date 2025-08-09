import dotenv
import traceback
import hydra
from omegaconf import DictConfig
import logging

from src.startup import RunBuilder

dotenv.load_dotenv(".env")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


@hydra.main(config_path="./config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    log.info(f"cfg.main: {cfg.main}")
    runner_cls = RunBuilder.get(cfg.main)
    runner = runner_cls(cfg)
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"exception arg: {e.args}")
        log.exception(f"traceback info: \n{traceback.format_exc()}")
