# shmr

Functions and analysis scripts for fitting fancy hypermutation modelswith ABC and neural nets.

As of this writing, the main files are
- `abc-functions.R`: Contains most of the functions for the ABC/net part. The two important ones are `get_net_summary_stats`, which computes summary statistics from a neural net and a set of sequences, and `check_abc_on_sims`, which goes through 

- `predictor-creation-functions.R`: A set of functions that create numeric predictors from sequences. To be used as input to neural nets.

- `shmr-functions.R`: A bunch of functions that I've used for other sorts of analysis, could be useful later, but not directly relevant to the neural net/ABC part of the project.

- `mmr-tests.R`: I barely remember doing this, but it seems to be something about estimating exo stripping parameters in the situation where you have AID lesions close together and the exo stripping can resolve one of the lesions. I don't think we're going to use this again, but it seemed to be working so I'm leaving it be for now.


In the analysis section, there are three scripts:

- `nnet-fits.R`: Fits and saves some of the nets I've been experimenting with. The fitted models are saved in `hdf5` format in the same directory.

- `nnet-trials.R`: Evaluates the nets for accuracy.

- `abc-trials.Rmd`: Uses the nets for ABC.



