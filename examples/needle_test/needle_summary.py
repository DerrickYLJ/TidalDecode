# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import Counter


def summary(run_name: str, output_path: str):
    pathlist = os.listdir(output_path)

    datas, cs = [], set()
    for path in pathlist:
        if run_name in path:
            data = json.load(open(output_path + "/" + path))
            if data[0]["context_length"] in cs:
                continue
            datas.extend(data)
            cs.add(data[0]["context_length"])

    res = Counter()
    for ii in datas:
        res[(ii["context_length"], ii["depth_percent"])] += ii["correct"] == True
        if ii["correct"] is False:
            print(ii["response"])
    sorted(res.items())
    with open(f"{output_path}/{run_name}.json", "w") as json_file:
        json.dump(datas, json_file)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--run_name", type=str, default=None)
    args.add_argument("--output_path", type=str, default="results/needle/")
    args.add_argument("--max_length", type=int, default=100000)
    args.add_argument("--min_length", type=int, default=1000)
    args.add_argument("--top_k", type=int, default=None)
    args = args.parse_args()
    args.output_path = os.path.join(
        args.output_path,
        str(args.top_k),
        f"{args.min_length//1000}K_{args.max_length//1000}K",
    )

    summary(
        run_name=args.run_name,
        output_path=args.output_path,
    )
