import argparse
import logging
import os
import pprint

import torch
import yaml
from bigmodelvis import Visualization
from easydict import EasyDict

from model.depictqa import DepictQA

from flask import Flask, request


def parse_args():
    parser = argparse.ArgumentParser(description="infer parameters for DepictQA")
    parser.add_argument("--cfg", type=str, default="./experiments/4kagent/config_comp.yaml")
    # infer cfgs, overwrite cfg.data.infer if set
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    return args

app = Flask(__name__)

@app.route("/compare_quality", methods=["POST"])
def query():
    """Compares the quality of two images.

    Args:
        imageA_path, imageB_path (Path)
        prompt (str)

    Returns:
        str: Response in text.
    """    
    assert request.form.keys() == {'imageA_path', 'imageB_path', 'prompt'}
    image_A = request.form.get('imageA_path')
    image_B = request.form.get('imageB_path')
    prompt = request.form.get('prompt')
    assert os.path.exists(image_A)
    assert os.path.exists(image_B)
    texts, output_ids, probs, confidences = model.generate(
        {
            "query": [prompt],
            "img_path": [None],
            "img_A_path": [image_A],
            "img_B_path": [image_B],
            "temperature": cfg.infer["temperature"],
            "top_p": cfg.infer["top_p"],
            "max_new_tokens": cfg.infer["max_new_tokens"],
            "task_type": "quality_compare_noref",
            "output_prob_id": cfg.infer["output_prob_id"],
            "output_confidence": cfg.infer["output_confidence"],
            "sentence_model": cfg.infer["sentence_model"],
        }
    )
    text = texts[0].strip()
    return {"answer": text}


if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    args = vars(args)
    args_filter = {}
    for key in args.keys():
        if key != "cfg" and args[key] is not None:
            args_filter[key] = args[key]
    # command line has higher priority
    cfg.data.infer.update(args_filter)

    logging.info("args: {}".format(pprint.pformat(cfg)))
    assert os.path.exists(
        cfg.model["vision_encoder_path"]
    ), "vision_encoder_path not exist!"
    assert os.path.exists(cfg.model["llm_path"]), "llm_path not exist!"
    assert os.path.exists(cfg.model["delta_path"]), "delta_path not exist!"

    # load model
    model = DepictQA(cfg, training=False)
    delta_ckpt = torch.load(cfg.model["delta_path"], map_location=torch.device("cpu"))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()
    Visualization(model).structure_graph()
    logging.info(f"[!] init the LLM over ...")

    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)