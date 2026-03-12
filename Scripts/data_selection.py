#!/usr/bin/env python3

import argparse as ap
import pandas as pd
import sys


def read_params():
    parser = ap.ArgumentParser(
        description="Select specific dataset from input dataset file"
    )
    arg = parser.add_argument
    arg(
        "inp_f",
        metavar="INPUT_FILE",
        nargs="?",
        default=sys.stdin,
        type=str,
        help="the input dataset file [stdin if not present]",
    )
    arg(
        "out_f",
        metavar="OUTPUT_FILE",
        nargs="?",
        default=None,
        type=str,
        help="the output dataset file",
    )

    arg("-z", "--feature_identifier", type=str, default="k__", help="the feature identifier")
    arg("-s", "--select", type=str, help="the samples to select")
    arg("-r", "--remove", type=str, help="the samples to remove")
    arg("-i", "--include", type=str, help="the fields to include")
    arg("-e", "--exclude", type=str, help="the fields to exclude")

    arg("-t", "--tout", action="store_true", help="transpose output dataset file")
    return vars(parser.parse_args())


def build_param_filters(par):
    frames = []
    if par["select"]:
        frames.append(
            pd.DataFrame(
                [s.split(":") for s in par["select"].split(",")],
                index=["select"] * (par["select"].count(",") + 1),
            )
        )
    if par["remove"]:
        frames.append(
            pd.DataFrame(
                [s.split(":") for s in par["remove"].split(",")],
                index=["remove"] * (par["remove"].count(",") + 1),
            )
        )
    if par["include"]:
        frames.append(
            pd.DataFrame(
                [s.split(":") for s in par["include"].split(",")],
                index=["include"] * (par["include"].count(",") + 1),
            )
        )
    if par["exclude"]:
        frames.append(
            pd.DataFrame(
                [s.split(":") for s in par["exclude"].split(",")],
                index=["exclude"] * (par["exclude"].count(",") + 1),
            )
        )
    return pd.concat(frames, axis=0) if frames else pd.DataFrame()


def main():
    par = read_params()

    f = pd.read_csv(par["inp_f"], sep="\t", header=None, index_col=0, dtype="unicode")
    f = f.T

    pf = build_param_filters(par)

    meta = [
        s
        for s in f.columns
        if sum([s2 in s for s2 in par["feature_identifier"].split(":")]) == 0
    ]
    if "unclassified" in meta:
        meta.remove("unclassified")
    feat = [
        s
        for s in f.columns
        if sum([s2 in s for s2 in par["feature_identifier"].split(":")]) > 0
    ]
    if "unclassified" in f.columns:
        feat.append("unclassified")

    for i in range(len(pf)):
        if pf.index[i] == "select":
            f = f[f[pf.iloc[i, 0]].isin(pf.iloc[i, 1:])]

        if pf.index[i] == "remove":
            f = f[~f[pf.iloc[i, 0]].isin(pf.iloc[i, 1:])]

        if pf.index[i] == "include":
            if pf.iloc[i, 0] != "feature_level":
                meta = [s for s in meta if s in pf.iloc[i, 0:].tolist()]
            else:
                feat = [s for s in feat if (pf.iloc[i, 1] in s) or ("unclassified" in s)]

        if pf.index[i] == "exclude":
            if pf.iloc[i, 0] != "feature_level":
                if pf.iloc[i, 0] == "_all_":
                    meta = []
                else:
                    meta = [s for s in meta if s not in pf.iloc[i, 0:].tolist()]
            else:
                if pf.iloc[i, 1] == "_all_":
                    feat = []
                else:
                    feat = [s for s in feat if pf.iloc[i, 1] not in s]

    f = f.loc[:, meta + feat]

    f.loc[:, feat] = f.loc[:, feat].replace(to_replace="nd", value="0.0")
    f.drop(
        f.loc[:, feat].columns[
            f.loc[:, feat].max().astype("float") == f.loc[:, feat].min().astype("float")
        ],
        axis=1,
        inplace=True,
    )

    if par["out_f"]:
        if par["tout"]:
            f.to_csv(par["out_f"], sep="\t", header=True, index=False, lineterminator="\n")
        else:
            f.T.to_csv(par["out_f"], sep="\t", header=False, index=True, lineterminator="\n")


if __name__ == "__main__":
    main()
