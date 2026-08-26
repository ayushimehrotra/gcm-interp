from ast import parse
import torch
import os
import random
import datetime
import time
import argparse
import sys
import yaml
from dataclasses import dataclass
import json
from eval.setup import set_seed
from eval.patch_site import SITES, SITE_OUT, dir_suffix, get_site
from eval.response_span import SPANS, SPAN_LEGACY, dir_suffix as span_suffix, get_span
from eval.localization_ctx import CTXS, CTX_DEFAULT, dir_suffix as ctx_suffix, get_ctx
class Config:
    def __init__(self):
        self.args = self.parse_arguments()


        self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/"

        if self.args.patch_algo == None:
            self.args.patch_algo = 'atp'
        self.setup_environment(seed=self.args.seed)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Patching')
        parser.add_argument('-d', '--device', type=str, default='cuda:1', required=True, help='Device to run the model on')
        parser.add_argument('-model_id', '--model_id', type=str, required=True, help='Model ID for the model')
        parser.add_argument('-batch_size', '--batch_size', type=int, default=8, required=True, help='Batch size for patching')
        parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('-ablation', '--ablation', type=str, default='steer', help='Apply steering ablation')
        parser.add_argument('-patch_model', '--patch_model', action='store_true', help='Patch the model')
        parser.add_argument('-eval_model', '--eval_model', action='store_true', help='Evaluate the model')
        parser.add_argument("--eval_test", nargs="?", const=True, default=None, help="Evaluate on test set. Optionally provide a test set name/path.")
        parser.add_argument('-eval_train', '--eval_train', action='store_true', help='Evaluate the model on train set')
        parser.add_argument('-eval_transfer', '--eval_transfer', type=str, help='Path to the test dataset for evaluation')
        parser.add_argument('--steering', action='store_true', help='Steering Eval mode')
        parser.add_argument('--pyreft', action='store_true', help='Use PyReFT Eval Mode')
        parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, default=256, help='Max new tokens to generate during eval')
        parser.add_argument('-patch_algo', '--patch_algo', type=str, help='acp/atp? acp for activation patching, atp for attribution patching')
        parser.add_argument('-patch_site', '--patch_site', type=str, default=SITE_OUT, choices=list(SITES),
                            help="Which tensor to score and steer (see eval/patch_site.py). "
                                 "'o_proj_out' (default) is o_proj.output, the attention block's "
                                 "contribution to the residual stream: W_O has already mixed the heads, "
                                 "so a block there is hidden_size//num_heads residual coordinates, not a "
                                 "head. 'o_proj_in' is o_proj.input, the concatenated per-head outputs, "
                                 "where block u IS head u and blocks are head_dim wide. The sites get "
                                 "separate results trees and their numbers are not comparable.")
        parser.add_argument('-response_span', '--response_span', type=str, default=SPAN_LEGACY, choices=list(SPANS),
                            help="Which tokens the localization metric scores (see eval/response_span.py). "
                                 "'legacy' (default, and every result in the paper) starts one position late "
                                 "and so never scores the FIRST response token -- for -single data that token "
                                 "is the entire answer, leaving only the turn-closing tokens. 'full' scores the "
                                 "whole response. 'full' writes to its own results tree; the two must not be mixed.")
        parser.add_argument('-localization_ctx', '--localization_ctx', type=str, default=CTX_DEFAULT,
                            choices=list(CTXS),
                            help="Which sequences supply the activations ATP differences (see "
                                 "eval/localization_ctx.py). net_effect = grad(A_base_full) * "
                                 "(A_src - A_base_patch); the gradient always comes from the "
                                 "response-bearing base, and only the two differenced tensors vary. "
                                 "'br-sq' (default, every result in the paper) differences a "
                                 "response-bearing base against a response-free source. 'br-sr' and "
                                 "'bq-sq' are the symmetric cells; 'bq-sr' the opposite asymmetry. "
                                 "Each writes to its own results tree.")
        parser.add_argument('-source', '--source', type=str, help='Patch from source')
        parser.add_argument('-base', '--base', type=str, help='Patch to base')
        parser.add_argument('-steering_add_path', '--steering_add_path', type=str, help='steering reps to add')
        parser.add_argument('-steering_sub_path', '--steering_sub_path', type=str, help='steering reps to subtract')
        parser.add_argument('--full_precision', action='store_true',
                            help='Load model in full bfloat16 with device_map=auto (no quantization). '
                                 'Required for very large models (e.g. 72B) that exceed single-GPU memory.')
        parser.add_argument('--kv_caching', action='store_true', help='Steer prefill only using KV cache; decoding steps are not re-steered')
        parser.add_argument('-steering_factors', '--steering_factors', type=str, default=None,
                            help='Comma-separated steering factors (N, the multiplier on the normalized '
                                 'steering vector) to sweep during --eval_model --steering. '
                                 'Defaults to the built-in sweep 1,2,4,5,6,8,10.')
        parser.add_argument('-topk_vals', '--topk_vals', type=str, default=None,
                            help='Comma-separated topk fractions to sweep during --eval_model --steering. '
                                 'Defaults to the built-in sweep 1.0,0.01,0.03,0.05,0.07,0.09,0.1,0.5.')
        parser.add_argument('--no_deterministic', action='store_true',
                            help='Disable the deterministic kernel settings (see determinism.py). '
                                 'Generation is nondeterministic without them -- two identical runs '
                                 'diverged in 24/50 continuations -- so arms generated with and '
                                 'without this flag are not comparable.')

        args = parser.parse_args()

        args.steering_factors = self._parse_number_list(args.steering_factors, parser, '--steering_factors')
        args.topk_vals = self._parse_number_list(args.topk_vals, parser, '--topk_vals')
        if not (args.patch_model or args.eval_model):
            parser.error("At least one of -patch_model, -eval_model is required")
        if args.patch_model or args.eval_model:
            if not args.patch_algo:
                parser.error("-patch_algo argument is required when --patch_model is set")
            if not args.source:
                parser.error("-source argument is required when --patch_model is set")
            if not args.base:
                parser.error("-base argument is required when --patch_model is set")

        if args.eval_model:
            if not args.eval_test:
                args.eval_train = True
            if isinstance(args.eval_test, str) and not os.path.exists(args.eval_test):
                parser.error(f"The provided eval_test path '{args.eval_test}' does not exist.")
            if isinstance(args.eval_test, str):
                args.test_dataset = args.eval_test.split('/')[-2]
                print(f"Steering dataset set to: {args.test_dataset}")
            elif isinstance(args.eval_test, bool) and args.steering:
                args.test_dataset = args.source

            if 'single' in args.test_dataset:
                # 24 (not 3): enough to reach the option letter when a model prefaces its
                # answer ("The answer is (B)"), which the scorer's parse_letter recovers.
                args.max_new_tokens = 24
            elif 'long' in args.test_dataset:
                args.max_new_tokens = 256
            
            args.steering_type = 'last_token'

        return args

    @staticmethod
    def _parse_number_list(raw, parser, flag_name):
        """'1,2,20' -> [1, 2, 20]; '0.5,2.5' -> [0.5, 2.5]. None passes through unchanged."""
        if raw is None:
            return None
        values = []
        for tok in str(raw).replace(' ', '').split(','):
            if not tok:
                continue
            try:
                values.append(float(tok) if '.' in tok else int(tok))
            except ValueError:
                parser.error(f"{flag_name} must be a comma-separated list of numbers, got '{tok}'")
        if not values:
            parser.error(f"{flag_name} was provided but empty")
        return values

    def save_to_yaml(self, file_path, args):
        args_dict = vars(args)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(args_dict, yaml_file, default_flow_style=False)
            
    def setup_environment(self, seed=42):
        set_seed(seed)

        os.makedirs(f'{self.set_output_prefix()}', exist_ok=True)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
        print(f'Saved config to file {self.get_output_prefix()}/config.yml')

    def get_output_prefix(self):
        return self.output_prefix
    
    def set_output_prefix(self):
        model = self.args.model_id.split('/')[-1]
        # The site is a suffix on the patch_algo DIRECTORY, not on
        # self.args.patch_algo itself: eval/logits_handler.py matches the arm off
        # that string (is_random/is_layer_matched/draw_seed) and must keep seeing
        # a bare 'random-s0'. Default site suffixes to '', so existing trees
        # resolve to exactly the paths they always have.
        algo_dir = (f"{self.args.patch_algo}{dir_suffix(get_site(self.args))}"
                    f"{span_suffix(get_span(self.args))}"
                    f"{ctx_suffix(get_ctx(self.args))}")
        eval_test_dir = self.args.eval_test.split('/')[-2] if isinstance(self.args.eval_test, str) else ''
        steering_dir = self.args.steering_add_path.split('/')[-2] if self.args.steering_add_path else ''
        if self.args.patch_model:
            self.output_prefix = f"./results/{model}/from_{self.args.source}_to_{self.args.base}/{algo_dir}/"
        if self.args.eval_model:
            self.output_prefix = f"./results/{model}/from_{self.args.source}_to_{self.args.base}/{algo_dir}/{eval_test_dir}_eval/{steering_dir}_steer/"
        print("op prefix ", self.output_prefix)
        return self.output_prefix
    
    def update_config(self, key, value):
        setattr(self.args, key, value)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
