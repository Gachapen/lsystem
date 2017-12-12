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

plot_population_comparison <- function(file) {
  stats <- read.csv(file = file, header = TRUE)
  stats <- stats[stats$score > 0, ]

  uniform <- stats[stats$population == 'uniform', 'score']
  sa <- stats[stats$population == 'sa', 'score']

  # ggplot(data = stats, aes(sample = score, group = population, color = population)) +
  #   geom_qq()

  # print(shapiro.test(uniform))
  # print(shapiro.test(sa))

  y <- c(uniform, sa)
  group <- as.factor(c(rep(1, length(uniform)), rep(2, length(sa))))
  print(levene.test(y, group))

  # print(t.test(uniform, sa, var.equal = TRUE))
  cat('Uniform median: ', median(uniform), ' (MAD: ', mad(uniform), ')\n')
  cat('Uniform mean: ', mean(uniform), ' (SD: ', sd(uniform), ')\n')
  cat('SA median: ', median(sa), ' (MAD: ', mad(sa), ')\n')
  cat('SA mean: ', mean(sa), ' (SD: ', sd(sa), ')\n')
  cat('Median diff: ', median(sa) - median(uniform), '\n')
  print(wilcox.test(uniform, sa))

  # print(var.test(uniform, sa))
  # print(t.test(uniform, sa, var.equal = TRUE))
  # print(wilcox.test(uniform, sa))

  ggplot(data=stats, aes(population, score)) +
    stat_summary(fun.y = median, geom = "bar") +
    stat_summary(fun.ymin = function(x) median(x) - mad(x), fun.ymax = function(x) median(x) + mad(x), geom = "errorbar") +
    scale_x_discrete(name="population") +
    scale_y_continuous(name="fitness median", limits=c(0, 1))
}

plot_score_distribution <- function(file, population) {
  data <- read.csv(file = file, header = TRUE)
  # population_data <- data[which(data$population == population), ]
  population_data <- data[which(data$population == population & data$score > 0), ]
  cat("Population size: ", length(population_data$score), " (", length(population_data$score) / length(data[which(data$population == population), ]$score) * 100, "%)\n")
  mean <- data.frame(label = "mean", val = mean(population_data$score, na.rm = T))
  median <- data.frame(label = "median", val = median(population_data$score, na.rm = T))
  averages <- rbind(mean, median)
  print(averages)

  ggplot(
    data = population_data,
    aes(
      x = score
    )
  ) +
    xlim(-0.05, 1.05) +
    geom_histogram(aes(y = ..density..), breaks = seq(0, 1, 0.01), colour="black", fill="white") +
    geom_density(alpha = 0.2, fill = "#FF6666") +
    geom_vline(data = averages, aes(xintercept = val, linetype = label, color = label), show.legend = TRUE)
}
