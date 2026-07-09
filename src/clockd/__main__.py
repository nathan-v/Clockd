import logging

import uvicorn

from clockd.config import load_server_config

if __name__ == "__main__":
    cfg = load_server_config()
    log_level = "info" if cfg.verbose else "warning"
    logging.basicConfig(level=getattr(logging, log_level.upper()), format="%(name)s %(message)s")
    uvicorn.run("clockd.main:app", host=cfg.host, port=cfg.port)
