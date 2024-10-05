# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

import numpy as np
import torch
from absl.app import run
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch.utils.checkpoint import checkpoint
from src.utils import load, download_url, load_jsonl


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class LLMNeedleHaystackTester:
    OURS_TEMPLATE = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n{context}\n\nQuestion: {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. Answer: "
    RANDOM_NEEDLE_CITIES = [
        "Chicago",
        "Yangon",
        "Antananarivo",
        "Colombo",
        "Almaty",
        "Sydney",
        "Chicago",
        "Mexico City",
        "Seattle",
        "Lagos",
        "Amsterdam",
        "Belgrade",
        "Cairo",
        "Baghdad",
        "Damascus",
        "Kigali",
        "Dakar",
        "Dakar",
        "Sofia",
        "Kigali",
        "Victoria",
        "Tashkent",
        "Mumbai",
        "Barcelona",
        "Almaty",
        "Amman",
        "Toronto",
        "Bratislava",
        "Johannesburg",
        "Thimphu",
        "Bangkok",
        "Santiago",
        "Cairo",
        "San Francisco",
        "Lagos",
        "Amsterdam",
        "Paris",
        "Rabat",
        "Santiago",
        "Copenhagen",
        "Madrid",
        "Kigali",
        "Ho Chi Minh City",
        "Sarajevo",
        "Delhi",
        "Istanbul",
        "Ho Chi Minh City",
        "Khartoum",
        "Helsinki",
        "Doha",
        "Istanbul",
        "Kuala Lumpur",
        "Budapest",
        "Shanghai",
        "Moscow",
        "Los Angeles",
        "Oslo",
        "Johannesburg",
        "Berlin",
        "Bangalore",
        "Tokyo",
        "Melbourne",
        "Barcelona",
        "Chicago",
        "Port Louis",
        "Lisbon",
        "Nairobi",
        "Kampala",
        "Lima",
        "Maputo",
        "Vancouver",
        "Dubai",
        "Khartoum",
        "Jakarta",
        "Madrid",
        "Yerevan",
        "Beirut",
        "Athens",
        "Chicago",
        "Paris",
        "Bucharest",
        "Copenhagen",
        "Brussels",
        "Damascus",
        "Seattle",
        "Los Angeles",
        "Yerevan",
        "Victoria",
        "Tunis",
        "Astana",
        "Seoul",
        "Buenos Aires",
        "Bangkok",
        "Colombo",
        "Brussels",
        "Khartoum",
        "Doha",
        "San Francisco",
        "Vienna",
        "Jakarta",
    ]

    def __init__(
        self,
        config,
        retrieval_question="What is the special magic {} number?",
        results_version=1,
        rnd_number_digits=7,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_interval_type="linear",
        save_results=False,
        final_context_length_buffer=200,
        print_ongoing_status=True,
        **kwargs,
    ):
        haystack_file = config.haystack_file
        context_lengths_min = config.context_lengths_min
        context_lengths_max = config.context_lengths_max
        context_lengths_num_intervals = config.n_context_length_intervals
        document_depth_percent_intervals = config.n_document_depth_intervals

        self.config = config
        self.needle = "\nThe special magic {city} number is: {rnd_number}\n"
        if not haystack_file or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )

        self.rnd_number_digits = rnd_number_digits
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.haystack_file = haystack_file
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        self.context_lengths = np.round(
            np.linspace(
                context_lengths_min,
                context_lengths_max,
                num=context_lengths_num_intervals,
                endpoint=True,
            )
        ).astype(int)
        if document_depth_percent_interval_type == "linear":
            self.document_depth_percents = np.round(
                np.linspace(
                    document_depth_percent_min,
                    document_depth_percent_max,
                    num=document_depth_percent_intervals,
                    endpoint=True,
                )
            ).astype(int)
        elif document_depth_percent_interval_type == "sigmoid":
            self.document_depth_percents = [
                self.logistic(x)
                for x in np.linspace(
                    document_depth_percent_min,
                    document_depth_percent_max,
                    document_depth_percent_intervals,
                )
            ]
        else:
            raise ValueError(
                f"Unsupported document_depth_percent_interval_type: {document_depth_percent_interval_type}"
            )
        if self.config.jobs is not None:
            start, end = self.config.jobs.split("-")
            print(self.context_lengths)
            self.context_lengths = self.context_lengths[int(start) : int(end)]
            print(self.context_lengths)
        self.model, self.tokenizer = load(config.model_name, config.attn_type, top_k=config.top_k, sparse_layer_start=config.sparse_layer_start,
            correction_layer=config.correction_layer)
        print(self.model)
        self.generation_config = GenerationConfig(
            max_new_tokens=32,
            pad_token_id=(
                self.tokenizer.pad_token_id if self.tokenizer != None else None
            ),
            eos_token_id=(
                self.tokenizer.eos_token_id if self.tokenizer != None else None
            ),
            do_sample=False,
        )

    def generate_random_number(self, num_digits):
        lower_bound = 10 ** (num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def generate_prompt(self, n_garbage, depth_ratio):
        """Generates a text file and inserts an execute line at a random position."""

        # Generate test depth
        # depth_ratio = 0 means random depth
        if depth_ratio == 0:
            n_garbage_prefix = random.randint(0, n_garbage)
        else:
            n_garbage_prefix = int(n_garbage * depth_ratio / 100)

        n_garbage_suffix = n_garbage - n_garbage_prefix
        task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
        garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        garbage_inf = " ".join([garbage] * 10000)
        assert len(garbage_inf) >= n_garbage
        garbage_prefix = garbage_inf[:n_garbage_prefix]
        garbage_suffix = garbage_inf[:n_garbage_suffix]
        pass_key = random.randint(1, 50000)
        information_line = (
            f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
        )
        final_question = "What is the pass key? The pass key is"
        lines = [
            task_description,
            garbage_prefix,
            information_line,
            garbage_suffix,
            final_question,
        ]
        return (
            "\n".join(lines),
            pass_key,
            "\n".join([task_description, garbage_prefix]),
            "\n".join([task_description, garbage_prefix, information_line]),
        )

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def read_context_files(self, n):
        max_context_length = max(self.context_lengths)
        contexts = []
        f = open(self.haystack_file, "r")
        for _ in range(n):
            context = ""
            toks = 0
            while toks < max_context_length:
                text = json.loads(f.readline())["text"]
                context += text
                toks += len(self.tokenizer.encode(text))
            contexts.append(context)
        return contexts

    def create_contexts(
        self,
        needle_rnd_number,
        insert_needle,
        random_city,
        trim_context,
        context_length,
        depth_percent,
        seed,
    ):
        needle = self.needle.format(city=random_city, rnd_number=needle_rnd_number)
        question = self.retrieval_question.format(random_city)
        if not insert_needle:
            needle = " "  # replace needle with a space
        context = self.insert_needle(
            needle, trim_context, depth_percent, context_length
        )
        results = {
            "context": context,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "needle": needle,
            "question": question,
            "insert_needle": insert_needle,
            "needle_rnd_number": needle_rnd_number,
            "seed": seed,
        }
        return results

    def insert_needle(self, needle, context, depth_percent, context_length):
        tokens_needle = self.tokenizer.encode(needle)
        tokens_context = self.tokenizer.encode(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.tokenizer.encode(".", add_special_tokens=False)

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_new_context)
        return new_context

    def run_test(self):
        contexts = []
        template = self.OURS_TEMPLATE

        def _key_from_result(result):
            return (result["context_length"], result["depth_percent"], result["seed"])

        results = []
        full_contexts = self.read_context_files(self.config.n_rounds)
        full_tokens = [
            self.tokenizer.encode(full_context) for full_context in tqdm(full_contexts)
        ]

        start = time.time()

        correct_cnt = 0
        total_cnt = 0
    
        for context_length in self.context_lengths:
            torch.cuda.empty_cache()
            trim_contexts = [
                self.tokenizer.decode(full_token[:context_length])
                for full_token in tqdm(full_tokens)
            ]
            contexts = []
            for depth_percent in self.document_depth_percents:
                for i in range(self.config.n_rounds):
                    random_city = random.choice(
                        LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES
                    )
                    insert_needle = True
                    needle_rnd_number = str(
                        self.generate_random_number(self.rnd_number_digits)
                    )
                    print("context length: " + str(context_length))
                    print("depth_percent : " + str(depth_percent))
                    context = self.create_contexts(
                        needle_rnd_number,
                        insert_needle,
                        random_city,
                        trim_contexts[i],
                        context_length,
                        depth_percent,
                        i,
                    )
                    contexts.append(context)

            for _, context in enumerate(tqdm(contexts)):
                depth = int(context["depth_percent"])
                length = context["context_length"]

                prompt = template.format(
                    context=context["context"], question=context["question"]
                )
                input_tensor = self.tokenizer(
                    prompt, return_tensors="pt", return_attention_mask=False
                )
                with torch.no_grad():
                    outs = self.model.generate(
                        **input_tensor,
                        generation_config=self.generation_config,
                        do_sample=False,
                    )
                new_tokens = outs[0, input_tensor["input_ids"].shape[-1] :]
                out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                init = time.time()
                results.append(
                    {
                        "context_length": context["context_length"],
                        "depth_percent": context["depth_percent"],
                        "response": out,
                        "answer": context["needle_rnd_number"],
                        "correct": context["needle_rnd_number"] in out,
                        "seed": context["seed"],
                    }
                )
                correct = context["needle_rnd_number"] in out
                correct_cnt = correct_cnt + 1 if correct else correct_cnt
                total_cnt += 1
                print(
                    f"depth: {depth/100}; len: {length}; inserted_pos: {int(depth*length//100)}: correct: {correct}", flush=True
                )
                print("output: ", out)
            with open(self.config.output_file, "w") as f:
                json.dump(results, f)
        print("elapsed", time.time() - start)
        print("done")
        print(
            f"correction_layer: {self.config.correction_layer}; top_k: {self.config.top_k}; correctness rate: {correct_cnt/total_cnt}"
        )

    def start_test(self):
        self.run_test()
