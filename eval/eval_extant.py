import matplotlib.pyplot as plt
import torch
import copy
import lm_eval
from lm_eval.evaluator import simple_evaluate
import json
import logging
import gc
print("LM EVAL VERSION")
# Set the logging level to WARNING to suppress DEBUG and INFO
logging.basicConfig(level=logging.WARNING)

from eval.patch_site import SITE_IN, get_site


class ExtantDatasetEvaluator:
    def __init__(self, model_handler, batch_handler, select_patch_activations, config, topk, topk_indices, op_path):
        if hasattr(model_handler, 'model') or hasattr(model_handler, 'tokenizer'):
            del model_handler.model
            del model_handler.tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        print('I am here!')
        """
        Initialize the ModelEvaluator with two models and their tokenizers.
        """
        self.config = config
        # if self.config.args.N == 1:
        #     self.config.args.N = self.eval_coh_dict[config.args.rm_id.split('/')[1]][config.args.source][topk]
        # assert self.config.args.N != 1, f"Topk {topk} not found in the eval_coh_dict for {config.args.rm_id.split('/')[1]} and {config.args.source}"
        self.topk_indices = topk_indices
        self.op_path = op_path
        self.batch_handler = batch_handler
        self.select_patch_activations = select_patch_activations
        self.model_handler = model_handler
        
        self.model1 = lm_eval.models.huggingface.HFLM(pretrained=self.config.args.model_id, device=self.config.args.device)
        self.model2 = lm_eval.models.huggingface.HFLM(pretrained=self.config.args.model_id, device=self.config.args.device)
        self.datasets = {
            "MMLU": {
                "task": "mmlu",
                "dataset_path": "cais/mmlu",
                "dataset_name": "all",
                "test_split": "test",
                "validation_split": "validation",
                "fewshot_split": "dev",
                "fewshot_config": {
                    "sampler": "first_n"
                },
                "output_type": "multiple_choice",
                "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
                "doc_to_choice": ["A", "B", "C", "D"],
                "doc_to_target": "answer",
                "metric_list": [{
                    "metric": "acc",
                    "aggregation": "mean",
                    "higher_is_better": "true"
                }],
                "metadata": {
                    "version": 1.0
                },
                "dataset_kwargs": {
                    "trust_remote_code": "true"
                }
            },
            # 'ARC-Easy': {
            #     "tag": ["ai2_arc"],
            #     "task": "arc_easy",
            #     "dataset_path": "allenai/ai2_arc",
            #     "dataset_name": "ARC-Easy",
            #     "output_type": "multiple_choice",
            #     "training_split": "train",
            #     "validation_split": "validation",
            #     "test_split": "test",
            #     "doc_to_text": "Question: {{question}}\nAnswer:",
            #     "doc_to_target": "{{choices.label.index(answerKey)}}",
            #     "doc_to_choice": "{{choices.text}}",
            #     "should_decontaminate": "true",
            #     "doc_to_decontamination_query": "Question: {{question}}\nAnswer:",
            #     "metric_list":[{ 
            #         "metric": "acc",
            #         "aggregation": "mean",
            #         "higher_is_better": "true"
            #     }],
            #     "metadata": {
            #         "version": 1.0   
            #     }
            # },
            # 'ARC-Challenge': {
            #     "tag": ["ai2_arc"],
            #     "task": "arc_challenge",
            #     "dataset_path": "allenai/ai2_arc",
            #     "dataset_name": "ARC-Challenge",
            #     "output_type": "multiple_choice",
            #     "training_split": "train",
            #     "validation_split": "validation",
            #     "test_split": "test",
            #     "doc_to_text": "Question: {{question}}\nAnswer:",
            #     "doc_to_target": "{{choices.label.index(answerKey)}}",
            #     "doc_to_choice": "{{choices.text}}",
            #     "should_decontaminate": "true",
            #     "doc_to_decontamination_query": "Question: {{question}}\nAnswer:",
            #     "metric_list":[{ 
            #         "metric": "acc",
            #         "aggregation": "mean",
            #         "higher_is_better": "true"
            #     }],
            #     "metadata": {
            #         "version": 1.0   
            #     }
            # }
        }
        
        accuracy_1 = self.get_accuracy(self.model1, edit=True)
        
        # TODO: LM-Eval breaks if this is not defined again. Not sure what the issue is.
        self.datasets = {
            "MMLU": {
                "task": "mmlu",
                "dataset_path": "cais/mmlu",
                "dataset_name": "all",
                "test_split": "test",
                "validation_split": "validation",
                "fewshot_split": "dev",
                "fewshot_config": {
                    "sampler": "first_n"
                },
                "output_type": "multiple_choice",
                "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
                "doc_to_choice": ["A", "B", "C", "D"],
                "doc_to_target": "answer",
                "metric_list": [{
                    "metric": "acc",
                    "aggregation": "mean",
                    "higher_is_better": "true"
                }],
                "metadata": {
                    "version": 1.0
                },
                "dataset_kwargs": {
                    "trust_remote_code": "true"
                }
            },
            # 'ARC-Easy': {
            #     "tag": ["ai2_arc"],
            #     "task": "arc_easy",
            #     "dataset_path": "allenai/ai2_arc",
            #     "dataset_name": "ARC-Easy",
            #     "output_type": "multiple_choice",
            #     "training_split": "train",
            #     "validation_split": "validation",
            #     "test_split": "test",
            #     "doc_to_text": "Question: {{question}}\nAnswer:",
            #     "doc_to_target": "{{choices.label.index(answerKey)}}",
            #     "doc_to_choice": "{{choices.text}}",
            #     "should_decontaminate": "true",
            #     "doc_to_decontamination_query": "Question: {{question}}\nAnswer:",
            #     "metric_list":[{ 
            #         "metric": "acc",
            #         "aggregation": "mean",
            #         "higher_is_better": "true"
            #     }],
            #     "metadata": {
            #         "version": 1.0   
            #     }
            # },
            # 'ARC-Challenge': {
            #     "tag": ["ai2_arc"],
            #     "task": "arc_challenge",
            #     "dataset_path": "allenai/ai2_arc",
            #     "dataset_name": "ARC-Challenge",
            #     "output_type": "multiple_choice",
            #     "training_split": "train",
            #     "validation_split": "validation",
            #     "test_split": "test",
            #     "doc_to_text": "Question: {{question}}\nAnswer:",
            #     "doc_to_target": "{{choices.label.index(answerKey)}}",
            #     "doc_to_choice": "{{choices.text}}",
            #     "should_decontaminate": "true",
            #     "doc_to_decontamination_query": "Question: {{question}}\nAnswer:",
            #     "metric_list":[{ 
            #         "metric": "acc",
            #         "aggregation": "mean",
            #         "higher_is_better": "true"
            #     }],
            #     "metadata": {
            #         "version": 1.0   
            #     }
            # }
        }
        accuracy_2 = self.get_accuracy(self.model2)

        with open(self.op_path, 'a') as f:
            json.dump({"edited": accuracy_1, "original": accuracy_2, "N": self.config.args.N, "topk": topk}, f)
            f.write('\n')
        # self._plot_accuracies(a ccuracy_1, accuracy_2)

        del self.model1
        del self.model2
        gc.collect()
        torch.cuda.empty_cache()
        
    def get_accuracy(self, model, batch_size=128, edit=False):
        """
        Compare the two models on all datasets and display a bar chart of their accuracies.
        """
        def edit_model():

            patch_activations = self.select_patch_activations.to(self.config.args.device)
            print('Patch Activations Shape: ', patch_activations.shape)
            DIM = self.model_handler.dim
            hook_handles = []
            # This path uses plain torch hooks, not nnsight, so the site picks the hook
            # type as well as the tensor: o_proj.output is a forward hook, o_proj.input a
            # forward PRE hook. Both edit the tensor in place, so neither has to return it.
            site = get_site(self.config.args)

            def _add_steering(tensor, layer_idx):
                head_indices = self.topk_indices[self.topk_indices['layer'] == layer_idx]['neuron'].unique()
                min_len = min(patch_activations.shape[1], tensor.shape[1])
                if layer_idx == 0:
                    print('Min Len: ', min_len)
                before_shape = tensor.shape
                for head_idx in head_indices:
                    head_idx = int(head_idx)
                    tensor[:, :min_len, DIM * head_idx : DIM * (head_idx + 1)] += \
                        self.config.args.N *patch_activations[layer_idx][:min_len, DIM * head_idx : DIM * (head_idx + 1)]
                assert before_shape == tensor.shape, f"Before shape {before_shape} and after shape {tensor.shape} do not match"
                return tensor

            def modify_generate_hook(layer_idx):
                def hook(module, input, output):
                    return _add_steering(output, layer_idx)
                return hook

            def modify_generate_pre_hook(layer_idx):
                def pre_hook(module, args):
                    _add_steering(args[0], layer_idx)
                return pre_hook

            # Register hooks for all relevant layers
            layer_indices = self.topk_indices['layer'].unique()
            for layer_idx in layer_indices:
                layer_idx = int(layer_idx)
                layer = model.model.model.layers[layer_idx]
                if site == SITE_IN:
                    handle = layer.self_attn.o_proj.register_forward_pre_hook(modify_generate_pre_hook(layer_idx))
                else:
                    handle = layer.self_attn.o_proj.register_forward_hook(modify_generate_hook(layer_idx))
                hook_handles.append(handle)
            return hook_handles
        
        if edit:
            hook_handles = edit_model()

        accuracies_model = {}
        limits = [16]
        for i, t in enumerate(self.datasets.keys()):
            task_config = copy.deepcopy(self.datasets[t])
            results_model = simple_evaluate(
                model=model,
                limit=limits[i],
                tasks=[task_config],
                batch_size=batch_size,
                num_fewshot=0,
                apply_chat_template=False,
                # log_samples=True,
            )["results"]
            print(self.datasets[t]['task'])
            result = {
                task: metrics["acc,none"] * 100
                for task, metrics in results_model.items()
                if "acc,none" in metrics and task == self.datasets[t]['task']
            }
            accuracies_model[self.datasets[t]['task']] = result[self.datasets[t]['task']]
        if edit:
            for handle in hook_handles:
                handle.remove()
                
        return accuracies_model
        

    def _plot_accuracies(self, accuracies_model1, accuracies_model2):
        """
        Plot a bar chart comparing the accuracies of the two models.
        """
        print("PLOT ACCURACIES ", accuracies_model1)
        print("PLOT ACCURACIES ", accuracies_model2)

        with open(self.op_path.replace('.png', '.json'), 'w') as f:
            json.dump({"edited": accuracies_model1, "original": accuracies_model2}, f)
        datasets = list(accuracies_model1.keys())
        model1_scores = [accuracies_model1[dataset] for dataset in datasets]
        model2_scores = [accuracies_model2[dataset] for dataset in datasets]

        x = range(len(datasets))
        width = 0.4

        plt.bar(x, model1_scores, width=width, label="Edited Model")
        plt.bar([p + width for p in x], model2_scores, width=width, label="Original Model")
        plt.xticks([p + width / 2 for p in x], datasets)
        plt.ylabel("Accuracy (%)")
        plt.title("Model Comparison on MMLU, ARC, and GSM8K")
        plt.legend()
        plt.savefig(self.op_path)
        plt.close()