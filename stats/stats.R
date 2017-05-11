require(ggplot2)
library(reshape)

stats <- read.csv(file = "stats.csv", header = TRUE)

# Calculate the percentage of available values (not NA) in a list x.
available_percentage <- function(x) {
  len <- length(x)
  num_na <- sum(is.na(x))
  percentage <- (len - num_na) / len * 100
  return(percentage)
}

percentage_something <- function(stats) {
  scores <- lapply(stats, "[[", "balance")
  success_rate <- sapply(scores, available_percentage)
  barplot(success_rate, ylim = c(0, 100), ylab = "% of something", xlab = "distribution")
}

se <- function(x) {
  return(sd(x, na.rm = TRUE) / sqrt(length(na.omit(x))))
}

score_changes <- function(stats) {
  mean_scores <- aggregate(stats, list(stats$distribution), mean, na.rm = TRUE)
  mean_scores <- mean_scores[2:8]
  mean_scores$drop <- -mean_scores$drop
  mean_scores$closeness <- -mean_scores$closeness
  mean_scores$nothing <- -mean_scores$nothing
  mean_scores <- melt(mean_scores, id = c("distribution"))

  scores_se <- aggregate(stats, list(stats$distribution), se)
  scores_se <- scores_se[2:8]
  scores_se <- melt(scores_se, id = c("distribution"))

  mean_scores$se <- scores_se$value

  ggplot(data = mean_scores, aes(x = distribution, y = value, colour = variable)) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = value - se, ymax = value + se), width = 0.1, alpha = 0.5) +
    ylim(-1.01, 1.01)
}
