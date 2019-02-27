#' Main function for fitting model by ABC
#'
#' @param sim_params Parameters used in the simulator.
#' @param sim_statistics Statistics corresponding to each parameter set.
#' @param obs_statistics Statistics for the observed data.
#' @param match_tol What fraction of the samples to keep.
#'
#' @return A matrix, columns representing parameters, of accepted parameters.
abc_rej <- function(sim_params, sim_statistics, obs_statistics, match_tol = .01) {
    obs_sim_dists = apply(sim_statistics, 1, function(x) sum((x - obs_statistics)^2))
    threshold = quantile(obs_sim_dists, probs = match_tol)
    return(sim_params[which(obs_sim_dists <= threshold),])
}
