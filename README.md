An attempt at replicating Bookworm Game: Automatic Discovery of LTE Vulnerabilities Through Documentation Analysis by Chen et al., doi: 10.1109/SP40001.2021.00104

The RODExtender kind of works, but:
- demo.txt is only a placeholder and does not return any RODs
- models and data have to be downloaded for this to work
- I have to trim down environment dependencies (CiRA & SPEC5G dependencies + spacy, ...)
- most if not all paths have to be modified
- I could clean up the code and the architecture by a lot and convert this into a pipeline
- the code from CiRA was not uploaded to git properly
