import pdb
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPTokenizer

sys.path.append(".")
sys.path.append("..")

import constants
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from models.xti_attention_processor import XTIAttenProc
from checkpoint_handler import CheckpointHandler
from utils import vis_utils
import os

import pidfile

import pickle


@dataclass
class InferenceConfig:
    name: Optional[str] = None
    diverse: Optional[bool] = False
    n: Optional[int] = 1000
    # Specifies which checkpoint iteration we want to load
    iteration: Optional[int] = None
    # The input directory containing the saved models and embeddings
    input_dir: Optional[Path] = None
    # Where the save the inference results to
    inference_dir: Optional[Path] = None
    # Specific path to the mapper you want to load, overrides `input_dir`
    mapper_checkpoint_path: Optional[Path] = None
    # Specific path to the embeddings you want to load, overrides `input_dir`
    learned_embeds_path: Optional[Path] = None
    # List of random seeds to run on
    seeds: List[int] = field(default_factory=lambda: [42])
    # If you want to run with dropout at inference time, this specifies the truncation indices for applying dropout.
    # None indicates that no dropout will be performed. If a list of indices is provided, will run all indices.
    truncation_idx = None
    # Whether to run with torch.float16 or torch.float32
    torch_dtype: str = "fp16"
    debug: Optional[bool] = False
    num_images_per_prompt: Optional[int] = 1
    gs: float = 6.5
    cls: str = ""

    def __post_init__(self):
        self._set_prompts()
        self._set_input_paths()
        self.inference_dir.mkdir(exist_ok=True, parents=True)
        self.torch_dtype = torch.float16 if self.torch_dtype == "fp16" else torch.float32

    def _set_input_paths(self):
        if self.inference_dir is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify inference_dir"
            self.inference_dir = self.input_dir / f"inference_{self.iteration}_{self.name}"
        if self.mapper_checkpoint_path is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify mapper_checkpoint_path"
            self.mapper_checkpoint_path = self.input_dir / f"mapper-steps-{self.iteration}.pt"
        if self.learned_embeds_path is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify learned_embeds_path"
            self.learned_embeds_path = self.input_dir / f"learned_embeds-steps-{self.iteration}.bin"

    def _set_prompts(self):
        pass


def load_stable_diffusion_model(pretrained_model_name_or_path: str,
                                learned_embeds_path: Path,
                                mapper: Optional[NeTIMapper] = None,
                                num_denoising_steps: int = 50,
                                torch_dtype: torch.dtype = torch.float16) -> Tuple[StableDiffusionPipeline, str, int]:
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = NeTICLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype,
    )
    if mapper is not None:
        text_encoder.text_model.embeddings.set_mapper(mapper)
    placeholder_token, placeholder_token_id = CheckpointHandler.load_learned_embed_in_clip(
        learned_embeds_path=learned_embeds_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    ).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_denoising_steps, device=pipeline.device)
    pipeline.unet.set_attn_processor(XTIAttenProc())
    return pipeline, placeholder_token, placeholder_token_id

def run_inference(prompt: str,
                  pipeline: StableDiffusionPipeline,
                  prompt_manager: PromptManager,
                  seed: int,
                  output_path: Optional[Path] = None,
                  num_images_per_prompt: int = 1,
                  truncation_idx: Optional[int] = None,
                 gs=7.0) -> Image.Image:
    with torch.autocast("cuda"):
        with torch.no_grad():
            prompt_embeds = prompt_manager.embed_prompt(prompt,
                                                        num_images_per_prompt=num_images_per_prompt,
                                                        truncation_idx=truncation_idx)
    joined_images = []
    generator = torch.Generator(device='cuda').manual_seed(seed)
    images = sd_pipeline_call(pipeline,
                              prompt_embeds=prompt_embeds,
                              generator=generator,
                              num_images_per_prompt=num_images_per_prompt,
                             guidance_scale=gs,
                             num_inference_steps=50).images
    # seed_image = Image.fromarray(np.concatenate(images, axis=1))
    #     save_name = f'{seed}_truncation_{truncation_idx}.png' if truncation_idx is not None else f'{seed}.png'
    #     seed_image.save(output_path / save_name)
#     joined_images.append(seed_image)
#     joined_image = vis_utils.get_image_grid(joined_images)
    return images

import matplotlib.pyplot as plt


def display_images_in_grid(image_path, rows, cols, figsize=(10, 10), titles=None):
    image_paths = [os.path.join(image_path, x) for x in os.listdir(image_path)]
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()

    for i in range(min(len(image_paths), rows * cols)):
        img_path = image_paths[i]
        img = Image.open(img_path)

        if titles is not None:
            axes[i].set_title(titles[i])

        axes[i].imshow(img)
        axes[i].axis('off')

    for i in range(len(image_paths), rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


@pyrallis.wrap()
def main(infer_cfg: InferenceConfig):
    if infer_cfg.diverse:
        import json
        with open('/data/vision/phillipi/perception/code/personalized-synthetic/notebooks/gpt4_prompts_proc.json', 'r') as f:
            diverse_prompts = json.load(f)
            print('USING GENERATED PROMPTS')
    else:
        print("USING VANILLA PROMPT")
    output_path = infer_cfg.inference_dir
    output_path.mkdir(exist_ok=True, parents=True)

    if len(os.listdir(output_path)) >= 1000:
        print("DONE ALREADY")
        sys.exit()

    if not infer_cfg.debug:
        pidfile.exit_if_job_done(output_path)

    train_cfg, mapper = CheckpointHandler.load_mapper(infer_cfg.mapper_checkpoint_path)
    pipeline, placeholder_token, placeholder_token_id = load_stable_diffusion_model(
        pretrained_model_name_or_path=train_cfg.model.pretrained_model_name_or_path,
        mapper=mapper,
        learned_embeds_path=infer_cfg.learned_embeds_path,
        torch_dtype=infer_cfg.torch_dtype,
        num_denoising_steps=50
    )
    prompt_manager = PromptManager(tokenizer=pipeline.tokenizer,
                                   text_encoder=pipeline.text_encoder,
                                   timesteps=pipeline.scheduler.timesteps,
                                   unet_layers=constants.UNET_LAYERS,
                                   placeholder_token=placeholder_token,
                                   placeholder_token_id=placeholder_token_id,
                                   torch_dtype=infer_cfg.torch_dtype)

    if not infer_cfg.diverse:
        for i in range(infer_cfg.n // infer_cfg.num_images_per_prompt):
            prompt = "A photo of a {}"
            seed = np.random.randint(1, sys.maxsize)
            image = run_inference(prompt=prompt,
                                  pipeline=pipeline,
                                  prompt_manager=prompt_manager,
                                  seed=seed,
                                  num_images_per_prompt=infer_cfg.num_images_per_prompt,
                                  truncation_idx=infer_cfg.truncation_idx,
                                  gs=infer_cfg.gs)

            for j, im in enumerate(image):
                save_name = f"{i*infer_cfg.num_images_per_prompt+j}.jpg"
                im.save(infer_cfg.inference_dir / save_name)
    else:
        diverse_prompts = diverse_prompts[infer_cfg.cls]
        for i in range(infer_cfg.n // infer_cfg.num_images_per_prompt):
            prompt = diverse_prompts[i % len(diverse_prompts)]
            seed = np.random.randint(1, sys.maxsize)
            try:
                image = run_inference(prompt=prompt,
                                      pipeline=pipeline,
                                      prompt_manager=prompt_manager,
                                      seed=seed,
                                      num_images_per_prompt=infer_cfg.num_images_per_prompt,
                                      truncation_idx=infer_cfg.truncation_idx,
                                      gs=infer_cfg.gs)
                for j, im in enumerate(image):
                    save_name = f"{i * infer_cfg.num_images_per_prompt + j}_{'_'.join(prompt.split())}.jpg"
                    im.save(infer_cfg.inference_dir / save_name)
            except Exception as e:
                print(f"Problem with {prompt}", e)

    if not infer_cfg.debug:
        pidfile.mark_job_done(output_path)

if __name__ == '__main__':
    main()