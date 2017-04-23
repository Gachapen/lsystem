stats <- read.csv(file = "stats.csv", header = TRUE)

stats <- split(stats, stats$distribution)  # Create list with each distribution
stats <- lapply(stats, subset, select = -distribution)  # Remove distribution variable

distribution_names <- c("uniform", "something", "zero")

# Calculate the percentage of available values (not NA) in a list x.
available_percentage <- function(x) {
  len <- length(x)
  num_na <- sum(is.na(x))
  percentage <- (len - num_na) / len * 100
  return(percentage)
}

percentage_something <- function(stats) {
  scores <- lapply(stats, "[[", "score")
  success_rate <- sapply(scores, available_percentage)
  names(success_rate) <- distribution_names
  barplot(success_rate, ylim = c(0, 4), ylab = "% of something", xlab = "distribution")
}

score_changes <- function(stats) {
  mean_scores <- lapply(stats, lapply, mean, na.rm = TRUE)
  mean_scores <- matrix(unlist(mean_scores), ncol = 3)  # transform to matrix
  colnames(mean_scores) <- c(0, 1, 2)
  rownames(mean_scores) <- c("score", "balance", "branching", "closeness", "drop")

  # Negate closenss and drop scores as they are punishments, not rewards.
  mean_scores["closeness", ] <- -mean_scores["closeness", ]
  mean_scores["drop", ] <- -mean_scores["drop", ]

  num_colors <- 5
  colors <- rainbow(num_colors)
  pch <- 21:25
  lty <- 1:5
  plot(0, 0, xlim = c(0, 2), ylim = c(-1.5, 1), ylab = "score", xlab = "distribution level",
    type = "n", xaxt = "n")
  for (i in 1:5) {
    lines(c(0, 1, 2), mean_scores[i, ], type = "o", col = colors[i], pch = pch[i],
      lty = lty[i])
  }
  axis(1, at = 0:2, labels = distribution_names)
  legend(0, 1, rownames(mean_scores), col = colors, cex = 0.8, pch = pch, lty = lty)
}
