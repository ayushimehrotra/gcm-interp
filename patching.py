import torch
import gc
from patching_utils import PatchingUtils
from eval.localization_ctx import get_ctx, base_has_response, source_has_response
import einops
import gc
class Patching:
    def __init__(self, model_handler, batch_handler, config):
        self.model_handler = model_handler
        self.batch_handler = batch_handler
        self.config = config
        self.print_now = False
        self.patching_utils = PatchingUtils(self)
        self.align_toks = self.patching_utils.align_toks
    
    def apply_patching(self):
        base_toks = self.batch_handler.base_toks
        source_qs_toks = self.batch_handler.source_qs_toks
        base_qs_toks = self.batch_handler.base_qs_toks
        response_start_positions = self.batch_handler.response_start_positions
        ctx = get_ctx(self.config.args)
        model = self.model_handler.model
        if self.config.args.patch_algo == 'acp':
            base_desired_logits_post_patch = self.patching_utils.patch_heads(base_toks['desired'], source_qs_toks['desired'], response_start_positions['base']['desired'])

            base_undesired_logits_post_patch = self.patching_utils.patch_heads(base_toks['undesired'], source_qs_toks['undesired'], response_start_positions['base']['undesired'])
            
            base_desired_logits_pre_patch, _ = self.patching_utils.get_activations(base_toks['desired'], which_patch='heads', resp_start_positions=response_start_positions['base']['desired'])
            
            base_undesired_logits_pre_patch, _ = self.patching_utils.get_activations(base_toks['undesired'], which_patch='heads', resp_start_positions=response_start_positions['base']['undesired'])
            
            logits = torch.stack([
                base_desired_logits_post_patch, 
                base_undesired_logits_post_patch,
                base_desired_logits_pre_patch.expand(base_desired_logits_post_patch.shape).to(base_desired_logits_post_patch.device),
                base_undesired_logits_pre_patch.expand(base_undesired_logits_post_patch.shape).to(base_undesired_logits_post_patch.device)
            ], dim=0).detach().cpu()
            torch.cuda.empty_cache()
            gc.collect()
            return logits
        elif 'atp' in self.config.args.patch_algo:
            base_desired_logits, base_desired_attn = self.patching_utils.get_activations(base_toks['desired'], which_patch='heads', resp_start_positions=response_start_positions['base']['desired'], retain_grad=True, logit=True)
            
            base_undesired_logits, base_undesired_attn = self.patching_utils.get_activations(base_toks['undesired'], which_patch='heads', resp_start_positions=response_start_positions['base']['undesired'], retain_grad=True, logit=True)

            if self.config.args.patch_algo == 'atp-zero':
                source_q_des_attn = [torch.zeros_like(bda) for bda in base_desired_attn]
                source_q_undes_attn = [torch.zeros_like(bua) for bua in base_undesired_attn]
            else:
                # --localization_ctx picks which source sequence supplies A_src: the
                # prompt only (historical) or the prompt WITH its assistant response.
                # Both are aligned to the base at the assistant marker, so the
                # difference below is taken position-by-position against the base.
                src_toks = self.batch_handler.source_toks if source_has_response(ctx) else source_qs_toks
                assert src_toks is not None, (
                    f"localization_ctx {ctx!r} needs a response-bearing source but "
                    "DataHandler built none -- source_toks is None.")
                source_q_des_attn = self.patching_utils.get_activations(src_toks['desired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=True, base_toks=base_toks['desired'])
                source_q_undes_attn = self.patching_utils.get_activations(src_toks['undesired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=True, base_toks=base_toks['undesired'])

            # A_base_patch. Default is the base run itself (the expansion point, so
            # the difference is a genuine first-order patching estimate). Under a
            # 'bq-*' ctx it is a SEPARATE prompt-only forward pass, aligned to the
            # same positions -- a valid contrast, but no longer an estimate of a
            # base->source patching effect. See eval/localization_ctx.py.
            base_patch_des_attn = base_patch_undes_attn = None
            if not base_has_response(ctx):
                base_patch_des_attn = self.patching_utils.get_activations(base_qs_toks['desired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=True, base_toks=base_toks['desired'])
                base_patch_undes_attn = self.patching_utils.get_activations(base_qs_toks['undesired'], which_patch='heads', resp_start_positions=None, logit=False, align=True, retain_grad=True, base_toks=base_toks['undesired'])

            L = base_undesired_logits - base_desired_logits
            L.backward(retain_graph=True)
            attn_desired_effects = []
            attn_undesired_effects = []
            net_effects = []
            for idx in range(len(model.model.layers)):
                base_des_val = base_desired_attn[idx].value
                base_undes_val = base_undesired_attn[idx].value
                src_des_val = source_q_des_attn[idx] if isinstance(source_q_des_attn[idx], torch.Tensor) else source_q_des_attn[idx].value
                src_undes_val = source_q_undes_attn[idx] if isinstance(source_q_undes_attn[idx], torch.Tensor) else source_q_undes_attn[idx].value
                # The GRADIENT factor is always the response-bearing base run: it is
                # the only thing L.backward() differentiated. Only the subtrahend moves.
                if base_patch_des_attn is None:
                    base_patch_des_val, base_patch_undes_val = base_des_val, base_undes_val
                else:
                    base_patch_des_val = base_patch_des_attn[idx] if isinstance(base_patch_des_attn[idx], torch.Tensor) else base_patch_des_attn[idx].value
                    base_patch_undes_val = base_patch_undes_attn[idx] if isinstance(base_patch_undes_attn[idx], torch.Tensor) else base_patch_undes_attn[idx].value
                attn_desired_effects.append(
                    base_des_val.grad *
                    (src_des_val - base_patch_des_val)
                )
                attn_undesired_effects.append(
                    base_undes_val.grad *
                    (src_undes_val - base_patch_undes_val)
                )

                net_effects.append(attn_desired_effects[idx].sum(dim=1) + attn_undesired_effects[idx].sum(dim=1))
            net_effects = torch.stack([h for h in net_effects], dim=0).detach().cpu()
            gc.collect()
            torch.cuda.empty_cache()
            print('net_effects', net_effects.shape)
            return net_effects