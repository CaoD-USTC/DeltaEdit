import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
from tqdm import tqdm
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
    KnownsDataset
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from AlphaEdit.AlphaEdit_main import get_cov
from DeltaEdit import DELTAHyperParams, apply_delta_to_model
from util import nethook
from util.globals import *


ALG_DICT = {
    "DeltaEdit": (DELTAHyperParams, apply_delta_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    use_save: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    file_handler = logging.FileHandler(str(run_dir / "logger.log"))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    if use_save:
        save_path = run_dir / "save_dir"
        save_path.mkdir()
        hparams.save_path = save_path

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")
    if any(alg in alg_name for alg in ["AlphaEdit"]):
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        for i, layer in enumerate(hparams.layers):
            force_recompute = False
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            ).cpu()
            weight = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight")
            P[i,:,:] = get_project(hparams, cov, weight)
            cache_c[i,:,:] += cov
    
    if ds_name == "mcf":
        neighborhood_true = {}
        neighborhood_true_path = DATA_DIR / (ds_name + '_' + model.config._name_or_path.split("/")[-1] + '_neighborhood_true.json')
        if neighborhood_true_path.exists():
            with open(neighborhood_true_path, "r") as file:
                neighborhood_true = json.load(file)
        old_len = len(neighborhood_true)
        # Iterate through dataset
        for record_chunks in tqdm(chunks(ds, num_edits), total=len(ds)):
            for record in record_chunks:
                case_ids = str(record['case_id'])
                if neighborhood_true.get(case_ids) is None:
                    target_true = vanilla_generation(model, tok, record, "cuda")
                    neighborhood_true[case_ids] = target_true
                else:
                    target_true = neighborhood_true[case_ids]
                record["neighborhood_true"] = target_true
        if len(neighborhood_true) > old_len:
            with open(neighborhood_true_path, "w") as file:
                json.dump(neighborhood_true, file) 

    num_edited = 0
    for record_chunks in chunks(ds, num_edits):
        num_edited += num_edits
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT", "DIAG"]) else dict()

        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit"]):
            seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit"]) else dict()
            nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit"]) else dict()
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
            )
        else:
            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start
        print("Execution took", exec_time)

 
        if num_edited % 100 == 0 and "PostEdit" not in alg_name:
            layers = hparams.layers if len(hparams.layers) > 1 and hparams.layers[-1] != hparams.layers[-2] else hparams.layers[:-1]
            for layer in layers:
                weight_edited = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight")
                condition_number = torch.linalg.cond(weight_edited)
                logger.info(f"{num_edited} {layer} condition number {condition_number}")
                logger.info(f"{num_edited} {layer} weight norm {torch.linalg.norm(weight_edited)}")

        if (num_edited == 200 or num_edited % 500 == 0 or num_edited == dataset_size_limit):
            test_ds = ds[:num_edited]
            test_path = run_dir / str(num_edited)
            if not test_path.exists():
                test_path.mkdir()
            test_case_result_template = str(test_path / "{}_edits-case_{}.json")
            for record_chunks in chunks(test_ds, num_edits):
                for record in record_chunks:
                    # Evaluate new model
                    start = time()
                    gen_test_vars = [snips, vec]
                    out_file = Path(test_case_result_template.format(num_edits, record["case_id"]))
                    if out_file.exists():
                        print(f"Skipping {out_file}; already exists")
                        continue
                    case_ids = [record["case_id"] for record in record_chunks]
                    metrics = {
                        "case_id": record["case_id"],
                        "grouped_case_ids": case_ids,
                        "num_edits": num_edits,
                        "requested_rewrite": record["requested_rewrite"],
                        "time": exec_time,
                        "post": ds_eval_method(
                            edited_model,
                            tok,
                            record,
                            *(
                                gen_test_vars
                                if record["case_id"] % generation_test_interval == 0
                                else [None, None]
                            ),  # Only test generation every generation_test_interval cases
                        ),
                    }

                    # Dump metrics in .json
                    with open(out_file, "w") as f:
                        json.dump(metrics, f, indent=1)

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def get_project(hparams, cov, weight=None):
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))

    P = U[:, small_singular_indices] @ U[:, small_singular_indices].T

    return P

def vanilla_generation(model, tok, record, device):
    prompts = record["neighborhood_prompts"]
    target = " " + record["requested_rewrite"]["target_true"]["str"]
    results = []
    target_new_tokens = tok.encode(target, add_special_tokens=False)
    for prompt in prompts:
        prompt_tok = tok(
            prompt,
            return_tensors="pt",
        ).to(device)
        gen_token = model.generate(
            input_ids=prompt_tok['input_ids'],
            attention_mask=prompt_tok['attention_mask'],
            max_new_tokens=len(target_new_tokens),
            pad_token_id=tok.eos_token_id,
            use_cache=False,
            do_sample=False,
            eos_token_id=None
        )
        token_ids = gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):]
        results.append(token_ids)
    return results

def vanilla_generation_zsre(model, tok, record, device):
    prompts = record["neighborhood"]["prompt"]
    target_new_tokens = record["neighborhood"]["target"]
    prompt_tok = tok(
        prompts,
        return_tensors="pt",
    ).to(device)
    gen_token = model.generate(
        input_ids=prompt_tok['input_ids'],
        attention_mask=prompt_tok['attention_mask'],
        max_new_tokens=len(target_new_tokens),
        pad_token_id=tok.eos_token_id,
        use_cache=False,
        do_sample=False,
        eos_token_id=None
    )
    token_ids = gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):]
    assert len(token_ids) == len(target_new_tokens)
    return token_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        # choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        # choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--use_save",
        default=False,
        dest="use_save",
        action="store_true",
        help="Save delta",
    )
    parser.add_argument(
        "--downstream_eval",
        default=False,
        dest="downstream_eval",
        action="store_true",
        help="If we want to do sequential editing or not",
    )
    parser.set_defaults(skip_generation_tests=True, conserve_memory=False)
    args = parser.parse_args()
    args.skip_generation_tests = False

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        use_save=args.use_save,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )
