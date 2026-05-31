import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# TODO extract `load_debug_rollout_data`


# TODO: remove `self`
def save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := self.args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO may improve the format
        if evaluation:
            dump_data = dict(
                samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
            )
        else:
            dump_data = dict(
                samples=[sample.to_dict() for sample in data],
            )

        torch.save(dict(rollout_id=rollout_id, **dump_data), path)
