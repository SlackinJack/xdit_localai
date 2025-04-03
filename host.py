# https://github.com/xdit-project/xDiT/blob/1c31746e2f903e791bc2a41a0bc23614958e46cd/comfyui-xdit/host.py

import os
import time
import torch
import torch.distributed as dist
import pickle
import io
import logging
import base64
import torch.multiprocessing as mp

# from compel import Compel, ReturnedEmbeddingsType
# from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler

from PIL import Image
from flask import Flask, request, jsonify
from xfuser import (
    xFuserArgs,
    xFuserFluxPipeline,
    xFuserHunyuanDiTPipeline,
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserStableDiffusion3Pipeline,
)
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_data_parallel_world_size,
    get_runtime_state,
    get_world_group,
    is_dp_last_group,
)

app = Flask(__name__)

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False


def setup_logger():
    global logger
    rank = dist.get_rank()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized:
        return jsonify({"status": "initialized"}), 200
    else:
        return jsonify({"status": "initializing"}), 202


def remove_extended_arg(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

extended_args = {}

def initialize():
    global pipe, engine_config, input_config, local_rank, initialized
    mp.set_start_method("spawn", force=True)

    parser = FlexibleArgumentParser(description="xFuser Arguments")

    # Extended arguments list
    global extended_args
    extended_args = {}
    parser.add_argument("--variant", type=str, default="fp16", help="PyTorch variant [fp16/fp32]")
    parser.add_argument("--pipeline_type", type=str, default=None, help="Pipeline type")
    # parser.add_argument("--compel", action="store_true", help="Enable Compel")
    # parser.add_argument("--scheduler", type=str, default="dpmpp_2m", help="Pipeline scheduler")
    extended_parser, xparser_args = parser.parse_known_args()

    # Remove extended arguments before passing to xFuserArgs
    if extended_parser.variant is not None:
        extended_args["variant"] = extended_parser.variant
        remove_extended_arg(parser, "variant")
    if extended_parser.pipeline_type is not None:
        extended_args["pipeline_type"] = extended_parser.pipeline_type
        remove_extended_arg(parser, "pipeline_type")
    else:
        assert False, "Pipeline type must be defined"
    # if extended_parser.compel is not None:
    #     extended_args["compel"] = extended_parser.compel
    #     remove_extended_arg(parser, "compel")
    # if extended_parser.scheduler is not None:
    #     extended_args["scheduler"] = extended_parser.scheduler
    #     remove_extended_arg(parser, "scheduler")

    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    setup_logger()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    pipeline_type = extended_args["pipeline_type"]
    pipeline_map = {
        "flux":         xFuserFluxPipeline,
        "hy":           xFuserHunyuanDiTPipeline,
        "pixart_alpha": xFuserPixArtAlphaPipeline,
        "pixart_sigma": xFuserPixArtSigmaPipeline,
        "sd3":          xFuserStableDiffusion3Pipeline,
    }
    PipelineClass = pipeline_map.get(pipeline_type)
    if PipelineClass is None:
        raise NotImplementedError(f"{pipeline_type} pipeline type not found")

    # scheduler = None
    # match extended_args["scheduler"]:
    #     case "ddim":        scheduler = DDIMScheduler
    #     case "dpmpp_2m":    scheduler = DPMSolverMultistepScheduler
    #     case "euler":       scheduler = EulerDiscreteScheduler
    #     case _:             raise NotImplementedError("{extended_args['scheduler']} is currently not supported!")
    # scheduler = scheduler.from_pretrained(engine_config.model_config.model, subfolder="scheduler")

    pipe = PipelineClass.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16 if extended_args['variant'] == "fp16" else torch.float32,
        # scheduler=scheduler,
        use_safetensors=True,
    ).to(f"cuda:{local_rank}")

    pipe.prepare_run(input_config)
    logger.info("Model initialization completed")
    initialized = True


def generate_image_parallel(
    positive_prompt, negative_prompt, num_inference_steps, seed, cfg, clip_skip
):
    global pipe, local_rank, input_config
    logger.info(f"Starting image generation with prompt: {positive_prompt}")
    logger.info(f"Negative: {negative_prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # positive_embeds = None
    # positive_pooled_embeds = None
    # negative_embeds = None
    # negative_pooled_embeds = None
    # if extended_args['compel']:
    #     compel = Compel(
    #         tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    #         text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    #         returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    #         requires_pooled=[False, True],
    #     )
    #     positive_embeds, positive_pooled_embeds = compel([positive_prompt])
    #     if negative_prompt and len(negative_prompt) > 0:
    #         negative_embeds, negative_pooled_embeds = compel([negative_prompt])

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=positive_prompt, # if positive_embeds is None else None,
        negative_prompt=negative_prompt, # if negative_embeds is None else None,
        num_inference_steps=num_inference_steps,
        output_type="pil",
        generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(seed),
        guidance_scale=cfg,
        clip_skip=clip_skip,
        # prompt_embeds=positive_embeds,
        # pooled_prompt_embeds=positive_pooled_embeds,
        # negative_embeds=negative_embeds,
        # negative_pooled_embeds=negative_pooled_embeds,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if is_dp_last_group():
        # serialize output object
        output_bytes = pickle.dumps(output)

        # send output to rank 0
        dist.send(torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0)
        dist.send(torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0)

        logger.info(f"Output sent to rank 0")

    if dist.get_rank() == 0:
        # recv from rank world_size - 1
        size = torch.tensor(0, device=f"cuda:{local_rank}")
        dist.recv(size, src=dist.get_world_size() - 1)
        output_bytes = torch.ByteTensor(size.item()).to(f"cuda:{local_rank}")
        dist.recv(output_bytes, src=dist.get_world_size() - 1)

        # deserialize output object
        output = pickle.loads(output_bytes.cpu().numpy().tobytes())

    return output, elapsed_time


@app.route("/generate", methods=["POST"])
def generate_image():
    logger.info("Received POST request for image generation")
    data = request.json
    positive_prompt     = data.get("positive_prompt", None)
    negative_prompt     = data.get("negative_prompt", None)
    num_inference_steps = data.get("num_inference_steps", 30)
    seed                = data.get("seed", 1)
    cfg                 = data.get("cfg", 3.5)
    clip_skip           = data.get("clip_skip", 0)

    logger.info(
        "Request parameters:\n"
        f"positive_prompt='{positive_prompt}'\n"
        f"negative_prompt='{negative_prompt}'\n"
        f"steps={num_inference_steps}\n"
        f"seed={seed}\n"
        f"cfg={cfg}\n"
        f"clip_skip={clip_skip}"
    )
    # Broadcast request parameters to all processes
    params = [positive_prompt, negative_prompt, num_inference_steps, seed, cfg, clip_skip]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output, elapsed_time = generate_image_parallel(*params)

    # Ensure output is not None before accessing its attributes
    if output and hasattr(output, "images") and output.images:
        pickled_image = pickle.dumps(output)
        output_base64 = base64.b64encode(pickled_image).decode('utf-8')
    else:
        output_base64 = ""
    image_path = ""

    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
        "output": output_base64,
    }
    return jsonify(response)


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="0.0.0.0", port=6000)
    else:
        while True:
            params = [None] * 6 # len(params) of generate_image_parallel()
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()

    logger.info(
        f"Process initialized. Rank: {dist.get_rank()}, Local Rank: {os.environ.get('LOCAL_RANK', 'Not Set')}"
    )
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    run_host()
