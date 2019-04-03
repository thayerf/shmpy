#' Sample MMR tract sizes
#'
#' Given a distribution for EXO stripping on either side and a
#' distance between AID lesions, use rejection sampling to sample from
#' the conditional distribution given that the amount of exo stripping
#' resulted in a single tract.
#'
#' @param n The number of samples.
#' @param p The parameter for the geometric distribution.
#' @param b The distance between the AID lesions.
#' @return A data frame with columns corresponding to a, b, c
#'     (distance to the left of the lesions, distance between lesions,
#'     distance to the right of th elesions) and rows corresponding to
#'     samples.
#' @export
generate_mmr_samples <- function(n, p, b) {
    out = matrix(nrow = n, ncol = 3)
    row = 1
    while(TRUE) {
        x = rgeom(4, p) + 1
        overlap = (x[2] + x[3]) >= b
        no_overwrite = x[2] < b
        if(overlap & no_overwrite) {
            a = max(x[1], x[3] - b)
            c = max(x[4], x[2] - b)
            out[row,] = c(a, b, c)
            row = row + 1
        }
        if(row > n) {
            break
        }
    }
    return(out)
}

likelihood_mmr_tract <- function(a, b, c, p) {
    l = (1 - (1 - p)^(b-1)) * (1 - (1 - p)^(a+b)) * (p * (1 - p)^(c-1)) * (p * (1 - p)^(a-1)) +
        (1 - (1 - p)^(b-1)) * (p * (1 - p)^(a+b-1)) * (p * (1 - p)^(c-1)) * (1 - (1-p)^(a-1))
    return(l)
}

#' 
test_likelihood <- function(n, ptrue, b) {
    data = generate_mmr_samples(n, ptrue, b)
    pvec = seq(0, 1, length.out = 1000)[2:999]
    individual_likelihoods = apply(data, 1, function(x) likelihood_mmr_tract(x[1], x[2], x[3], pvec))
    ## the above gives one column for each sample, so we want to sum row by row
    ll = rowSums(log(individual_likelihoods)) / n
    return(data.frame(loglik = ll, p = pvec))
}


#' Sample MMR tract sizes
#'
#' Given a distribution for EXO stripping on either side and a
#' distance between AID lesions, use rejection sampling to sample from
#' the conditional distribution given that the amount of exo stripping
#' resulted in a single tract.
#'
#' @param n The number of samples.
#' @param pl The parameter for the geometric distribution for the left-hand stripping.
#' @param pr The parameter for the geometric distribution for the right-hand stripping.
#' @param b The distance between the AID lesions.
#' @return A data frame with columns corresponding to a, b, c
#'     (distance to the left of the lesions, distance between lesions,
#'     distance to the right of th elesions) and rows corresponding to
#'     samples.
#' @export
generate_mmr_samples_asymmetric <- function(n, pl, pr, b) {
    out = matrix(nrow = n, ncol = 3)
    row = 1
    while(TRUE) {
        x = c(rgeom(1, pl), rgeom(1, pr), rgeom(1, pl), rgeom(1, pr)) + 1
        first_repaired = sample(1:2, 1)
        overlap = (x[2] + x[3]) >= b
        if(first_repaired == 1) {
            no_overwrite = x[2] < b
        } else {
            no_overwrite = x[3] < b
        }
        if(overlap & no_overwrite) {
            a = max(x[1], x[3] - b)
            c = max(x[4], x[2] - b)
            out[row,] = c(a, b, c)
            row = row + 1
        }
        if(row > n) {
            break
        }
    }
    return(out)
}

#' 
likelihood_mmr_tract_asymmetric <- function(a, b, c, pl, pr) {
    l = (1 - (1 - pr)^(b-1)) * (1 - (1 - pl)^(a+b)) * (pr * (1 - pr)^(c-1)) * (pl * (1 - pl)^(a-1)) +
        (1 - (1 - pr)^(b-1)) * (pl * (1 - pl)^(a+b-1)) * (pr * (1 - pr)^(c-1)) * (1 - (1-pl)^(a-1)) +
        (1 - (1 - pl)^(b-1)) * (1 - (1 - pr)^(b + c)) * (pl * (1 - pl)^(a-1)) * (pr * (1 - pr)^(c-1)) +
        (1 - (1 - pl)^(b-1)) * (pr * (1 - pr)^(b+c-1)) * (pl * (1 - pl)^(a-1)) * (pr * (1 - pr)^(c-1))
    return(l)
}

#' 
test_likelihood_asymmetric <- function(n, pltrue, prtrue, b) {
    data = generate_mmr_samples_asymmetric(n, pltrue, prtrue, b)
    plvec = prvec = seq(0, 1, length.out = 100)[2:99]
    grid = expand.grid(pl = plvec, pr = prvec)
    ll = apply(grid, 1, function(p) {
        pl = p[1]
        pr = p[2]
        sum(log(apply(data, 1, function(x) likelihood_mmr_tract_asymmetric(x[1], x[2], x[3], pl, pr)))) / n
    })
    return(data.frame(loglik = ll, grid))
}

test_likelihood_optim <- function(n, pltrue, prtrue, b) {
    data = generate_mmr_samples_asymmetric(n, pltrue, prtrue, b)
    fn = function(params) {
        transformed_params = 1 / (1 + exp(-params))
        -sum(log(apply(data, 1, function(x) likelihood_mmr_tract_asymmetric(x[1], x[2], x[3], transformed_params[1], transformed_params[2]))) / n)
    }
    o = optim(c(.5, .5), fn)
    return(1 / (1 + exp(-o$par)))
}

