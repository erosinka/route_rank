# trip_rank

## description

Contains python script `main.py` to rank trips based on data provided in the input `json` file using the [PROMETHEE](https://en.wikipedia.org/wiki/Preference_ranking_organization_method_for_enrichment_evaluation) multi-criteria decision-making method.
Criteria are weighted using the data from the `data/weights.json` file that can be edited accordingly to the user's preferences. Field names in the `weights.json` file should be the same as in the input data.


## usage

```
$ python main.py [-h] filename [sorted]

```

where `filename` is a relative path to the input `json` file and `sorted` is an optional boolean argument. Its value is `False` by defaut, if `True` then the output data is sorted. The output saved to the `data/ex3-output.json`. Input file should not be empty, and trips are expected to be all _one-way_ or _return_.

## output

Ouput data is saved to `data/ouput.json` file containing the input data with fields `rank` and `score` added. Higher score corresponds to higher rank where 1 is the highert rank.
