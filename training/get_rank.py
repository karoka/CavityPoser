import pandas as pd
import sys
import os

import warnings
warnings.filterwarnings("ignore")

def get_rank(datahome, testid = 1):
    res = pd.read_csv("%s/test_set%d_results.csv" %(datahome, testid))
    res["rank"] = res.groupby(by="0")["pred_COMP"].rank("dense", ascending=False)
    res["rank_pct"] = res.groupby(by="0")["pred_COMP"].rank("dense", ascending=False, pct=True)
    res["rank"] = res["rank"].astype(int)
    rank_native = res[res["1"] == "native"]
    rank_native["count"] = res.groupby(by="0")["pred_COMP"].count().values
    rank_native['rank'] = rank_native["rank"].astype(str) + "/" + rank_native["count"].astype(str)
    return rank_native[rank_native["count"] != 1][["0", "rank", "rank_pct"]]

def main():
    testid = int(sys.argv[1])
    # resdir = sys.argv[2]
    resdir = "results"

    datahome = "%s/%s" %(os.getcwd(), resdir)
    rank = get_rank(datahome, testid = testid)
    rank.to_csv("%s/rank_testset%d.csv" %(datahome, testid), float_format='%.3f')

if __name__ == "__main__":
    main()
