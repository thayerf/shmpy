#' Creates a matrix representation of a sequence
#'
#' @param seq A string with the sequence.
#'
#' @return A matrix, number of rows equal to the length of the
#' sequence, 4 columns. Element (i, j)=1 indicates that the base at
#' the ith position in the sequence is the jth element of the vector
#' (A, T, G, C)
#' @export
sequence_to_matrix <- function(seq, nuc = c("A", "T", "G", "C")) {
    seq_matrix= matrix(0, nrow = nchar(seq), ncol = 4)
    s_vec = strsplit(seq, split = "", fixed = TRUE)[[1]]
    for(i in 1:length(s_vec)) {
        seq_matrix[i,which(nuc == s_vec[i])] = 1
    }
    rownames(seq_matrix) = s_vec
    colnames(seq_matrix) = c("A", "T", "G", "C")
    return(seq_matrix)
}


#' Creates an array of one-hot matrix representations of a set of
#' sequences.
#'
#' @param seqs A vector of strings, each one a sequence.
#'
#' @return An array, first dimension equal to the number of sequences,
#' second dimension equal to the length of each sequence, third
#' dimension equal to 4 (number of nucleotides). Element (i, j, k) = 1
#' indicates that the base at the jth position in the ith sequence is
#' the kth element of the vector (A, T, G, C).
#' @export
one_hot_2d_sequences <- function(seqs) {
    matrices = lapply(seqs, sequence_to_matrix)
    out = array(0, dim = c(length(seqs), dim(matrices[[1]])))
    for(i in 1:length(matrices)) {
        out[i,,] = matrices[[i]]
    }
    return(out)
}

#' Creates a matrix of one-hot representations of sequences.
#'
#' @param seqs A vector of strings, each one a sequence.
#'
#' @return A matrix, number of rows equal to the number of sequences,
#' number of columns four times the length of each sequence. Each row
#' is a vectorized version of the output from sequence_to_matrix.
#' @export
one_hot_1d_sequences <- function(seqs) {
    t(sapply(seqs, function(s) {
        as.vector(t(sequence_to_matrix(s)))
    }))
}

#' Creates an array of one-hot representations of sequences for use as
#' input to a convolutional net.
#'
#' @param seqs A vector of strings, each one a sequence.
#'
#' @return The same as one_hot_2d_sequences, but with one extra
#' dimension.
#' @export
one_hot_2d_sequences_for_conv_net <- function(seqs) {
    oh = one_hot_2d_sequences(seqs)
    dim(oh) = c(dim(oh), 1)
    return(oh)
}
