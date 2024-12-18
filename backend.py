#!/usr/bin/env python3
import argparse
import backend_pb2
import backend_pb2_grpc
import base64
import grpc
import json
import os
import pickle
import requests
import signal
import subprocess
import sys
import time
import torch
from concurrent import futures
from pathlib import Path
from PIL import Image


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
# COMPEL = os.environ.get("COMPEL", "0") == "1"
WARMUP_STEPS = 0
HOST_INIT_TIMEOUT = 90


# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))


process = None


def kill_process():
    global process
    if process is not None:
        process.terminate()
        time.sleep(15)
        process = None


# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    last_height = None
    last_width = None
    last_model_path = None
    last_cfg_scale = 7
    last_scheduler = None
    variant = "fp32"
    # last_lora_adapters = []
    # loras = {}
    last_clip_skip = 0
    is_low_vram = False

    is_loaded = False
    needs_reload = False


    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))


    def LoadModel(self, request, context):
        if request.CFGScale != 0 and self.last_cfg_scale != request.CFGScale:
            self.last_cfg_scale = request.CFGScale

        if request.Model != self.last_model_path or (
            len(request.ModelFile) > 0 and (
                os.path.exists(request.ModelFile) and (
                    request.ModelFile != self.last_model_path
                )
            )
        ):
            self.needs_reload = True

        self.last_model_path = request.Model
        if request.ModelFile != "":
            if os.path.exists(request.ModelFile):
                self.last_model_path = request.ModelFile

        if request.F16Memory and self.variant != "fp16":
            self.needs_reload = True
            self.variant = "fp16"
        elif not request.F16Memory and self.variant != "fp32":
            self.needs_reload = True
            self.variant = "fp32"

        # if request.SchedulerType != "" and request.SchedulerType != self.last_scheduler:
        #     self.needs_reload = True
        #     self.last_scheduler = request.SchedulerType

        # if request.LoraAdapters:
        #     if len(self.last_lora_adapters) == 0 and len(request.LoraAdapters) == 0:
        #         pass
        #     elif len(self.last_lora_adapters) != len(request.LoraAdapters):
        #         self.needs_reload = True
        #     else:
        #         for adapter in self.last_lora_adapters:
        #             if adapter not in request.LoraAdapters:
        #                 self.needs_reload = True
        #                 break
        #     if self.needs_reload:
        #         self.loras = {}
        #         self.last_lora_adapters = request.LoraAdapters
        #         if len(request.LoraAdapters) > 0:
        #             i = 0
        #             for adapter in request.LoraAdapters:
        #                 self.loras[adapter] = request.LoraScales[i]
        #                i += 1

        if request.CLIPSkip != self.last_clip_skip:
            self.last_clip_skip = request.CLIPSkip

        if self.is_low_vram != request.LowVRAM:
            self.needs_reload = True
            self.is_low_vram = request.LowVRAM

        return backend_pb2.Result(message="", success=True)


    def GenerateImage(self, request, context):
        if request.height != self.last_height or request.width != self.last_width:
            self.needs_reload = True

        if not self.is_loaded or self.needs_reload:
            kill_process()
            nproc_per_node = torch.cuda.device_count()
            self.last_height = request.height
            self.last_width = request.width

            # ensure the following is true:
            # (pipefusion * tensor * data * ulysses * ring) <= nproc_per_node
            pipefusion_parallel_degree = nproc_per_node
            tensor_parallel_degree = 1
            data_parallel_degree = 1
            ulysses_degree = 1
            ring_degree = 1
            CFG_ARGS=""

            # enable to use more VRAM, but faster
            # best to enable, if possible
            cfg_parallel = True
            if cfg_parallel: # and nproc_per_node % 2 == 0:
                pipefusion_parallel_degree = 1 # int(nproc_per_node / 2)
                CFG_ARGS="--use_cfg_parallel"

            # scheduler = self.last_scheduler
            # if scheduler is None or len(scheduler) == 0:
            #     scheduler = "dpmpp_2m"

            pipeline_type = None
            model_name = self.last_model_path.lower()
            if "pixart" in model_name and "xl" in model_name:
                pipeline_type = "pixart_alpha"
            elif "pixart" in model_name and "sigma" in model_name:
                pipeline_type = "pixart_sigma"
            elif ("stable" in model_name and "diffusion" in model_name and "3" in model_name) or ("sd3" in model_name):
                pipeline_type = "sd3"
            elif "hunyuan" in model_name:
                pipeline_type = "hy"
            elif "flux" in model_name:
                pipeline_type = "flux"
            else:
                kill_process()
                assert False, "Unknown model type"

            cmd = [
                'torchrun',
                f'--nproc_per_node={nproc_per_node}',
                'host.py',

                f'--model={self.last_model_path}',
                f'--pipeline_type={pipeline_type}',
                f'--variant={self.variant}',
                f'--height={self.last_height}',
                f'--width={self.last_width}',
                # f'--scheduler={scheduler}',
                f'--pipefusion_parallel_degree={pipefusion_parallel_degree}',
                f'--tensor_parallel_degree={tensor_parallel_degree}',
                f'--data_parallel_degree={data_parallel_degree}',
                f'--ulysses_degree={ulysses_degree}',
                f'--ring_degree={ring_degree}',
                f'--prompt="setup"',
                '--no_use_resolution_binning',    # needs further research into its effects
                '--use_cache',                    # needs further research into its effects
                f'--warmup_steps={WARMUP_STEPS}',
                CFG_ARGS,
            ]

            # if COMPEL:
            #     cmd.append('--compel')

            if self.is_low_vram:
                # cmd.append('--use_parallel_vae') # breaks generation
                cmd.append('--enable_model_cpu_offload')
                cmd.append('--enable_sequential_cpu_offload')
                cmd.append('--enable_tiling')
                cmd.append('--enable_slicing')

            cmd = [arg for arg in cmd if arg]
            global process
            process = subprocess.Popen(cmd)
            initialize_url = "http://localhost:6000/initialize"
            time_elapsed = 0
            while True:
                try:
                    response = requests.get(initialize_url)
                    if response.status_code == 200 and response.json().get("status") == "initialized":
                        self.is_loaded = True
                        self.needs_reload = False
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
                time_elapsed += 1
                if time_elapsed > HOST_INIT_TIMEOUT:
                    kill_process()
                    return backend_pb2.Result(message=f"Failed to launch host within {HOST_INIT_TIMEOUT} seconds", success=False)

        if self.is_loaded:
            url = 'http://localhost:6000/generate'
            data = {
                "positive_prompt": request.positive_prompt,
                "num_inference_steps": request.step,
                "seed": request.seed,
                "cfg": self.last_cfg_scale,
                "clip_skip": self.last_clip_skip,
            }

            if request.negative_prompt and len(request.negative_prompt) > 0:
                data["negative_prompt"] = request.negative_prompt

            response = requests.post(url, json=data)
            response_data = response.json()
            output_base64 = response_data.get("output", "")

            if output_base64:
                output_bytes = base64.b64decode(output_base64)
                output = pickle.loads(output_bytes)
            else:
                output = None
                kill_process()
                assert False, "No output object received"
            image = output.images[0]
            if image.size == (0,0) or not image.getbbox():
                return backend_pb2.Result(message="No image generated", success=False)
            else:
                image.save(request.dst)
                return backend_pb2.Result(message="Media generated", success=True)
        else:
            return backend_pb2.Result(message="Host is not loaded", success=False)


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    # Define the signal handler function
    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        kill_process()
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        kill_process()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
