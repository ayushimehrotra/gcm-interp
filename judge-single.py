import json
import os
steering_factors = [20, 15, 10, 8, 6, 5, 4, 2, 1]
topk_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]
model_name = "Qwen1.5-14B-Chat"
tasks = ["from_harmful-long_to_harmless", "from_harmful-single_to_harmless", "from_verse-long_to_prose", "from_verse-single_to_prose"]
methods = ['random']
base_dir = f"/mnt/align4_drive/arunas/multi-token/gcm-interp/results/{model_name}"

def calculate_accuracy(file_path, base):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    correct = 0
    total = 0
    for item in data:
        # print(base, item)
        if base == 'harmless':
            if 'no' in item[f'old_{base}'].lower():
                continue
            else:
                total += 1
                if 'no' in item[f'edit_{base}'].lower():
                    correct += 1
        elif base == 'prose':
            if 'verse' in item[f'old_{base}'].lower():
                continue
            else:
                total += 1
                if 'verse' in item[f'edit_{base}'].lower():
                    correct += 1
    print(f"Correct: {correct}, Total: {total}")
    return correct / total if total > 0 else 0

for task in tasks:
    base = task.split("_")[-1]
    source = task.split("_to_")[0].split("from_")[1].split("-")[0]
    task_dir = os.path.join(base_dir, task)
    for method in methods:
        method_dir = os.path.join(task_dir, method)
        for eval_dir, steering_dir in [('harmful-single_eval', 'harmful-long_steer'), ('harmful-single_eval', 'harmful-single_steer'), ('verse-single_eval', 'verse-long_steer'), ('verse-single_eval', 'verse-single_steer')]:
                for steering_factor in steering_factors:
                    for topk in topk_values:
                        try:
                            file_path = os.path.join(method_dir, eval_dir, steering_dir, "eval/", f"{steering_factor}_random_steer_{topk}_{source}-single_gen.json")
                            accuracy = calculate_accuracy(file_path, base)
                            os.makedirs(f'results/accuracy/{model_name}/{task}/{method}/{eval_dir}/{steering_dir}', exist_ok=True)
                            with open(f'results/accuracy/{model_name}/{task}/{method}/{eval_dir}/{steering_dir}/{steering_factor}_random_steer_topk_{topk}_gen_accuracy_w_rf.json.accuracy.json', 'w') as out_f:
                                json.dump({'q1': accuracy}, out_f)
                        except FileNotFoundError as e:
                            continue