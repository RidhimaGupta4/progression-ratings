# Third-party notices

## StatsBomb Open Data
All analysis in this repository is based on StatsBomb Open Data
(https://github.com/statsbomb/open-data, commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`), used under the
StatsBomb Public Data User Agreement (`LICENSE.pdf` in that repository).

- No StatsBomb data are included in this repository. `fetch_data.sh` downloads the files directly from
  StatsBomb's repository, and every file is checked against its git blob hash in `data_manifest.tsv`.
- The results in `results/`, the figures in `figures/` and the interactive page in `docs/` are derived
  analysis of StatsBomb data. They are shared for non-commercial use, as the agreement requires, and are not
  covered by this repository's MIT licence.
- The StatsBomb logo (`assets/statsbomb_logo.png`, `explorer/statsbomb_logo_small.png`, a resized copy) is
  taken unmodified from `img/SB - Icon Lockup - Colour positive.png` in the same repository and commit. It
  is shown to credit StatsBomb as the data source, as the agreement requires; it remains StatsBomb's
  trademark and is not covered by this repository's licence.
- Users of the data are asked by StatsBomb to register at https://statsbomb.com/resource-centre/.

## Barlow and Barlow Condensed fonts
Embedded in the interactive page. Copyright 2017 The Barlow Project Authors
(https://github.com/jpt/barlow), licensed under the SIL Open Font License, Version 1.1
(full text in `explorer/fonts/OFL.txt`).

## Method credit
The action-value model follows the VAEP framework of Decroos, Bransen, Van Haaren and Davis, "Actions Speak
Louder than Goals: Valuing Player Actions in Soccer", KDD 2019. It is an independent implementation; no code
from the authors' `socceraction` package is used.
