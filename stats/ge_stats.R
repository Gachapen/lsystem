library(ggplot2)
library(gridExtra)
library(scales)

plot_ge <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot() +
    ylim(0, 1) +
    geom_point(data = stats, aes(iteration, avg), size = 0.5, color = "blue3") +
    geom_point(data = stats, aes(iteration, best), size = 0.5, color = "green3")
}

plot_ge_comparison <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  random <- stats[stats$label == 'random', 'best']
  ge <- stats[stats$label == 'ge', 'best']

  print(var.test(random, ge))
  print(t.test(random, ge, var.equal = TRUE))
  print(wilcox.test(random, ge))

  ggplot(data=stats, aes(label, best)) +
    stat_summary(fun.y = mean, geom = "bar") +
    stat_summary(fun.data = mean_se, geom = "errorbar") +
    scale_x_discrete(name="method") +
    scale_y_continuous(name="mean of best fitness")
}

plot_ge_comparison_duration <- function(file) {
  stats <- read.csv(file = file, header = TRUE)
  stats[, 'duration'] <- stats[, 'duration'] / 11

  random <- stats[stats$label == 'random', 'duration']
  ge <- stats[stats$label == 'ge', 'duration']

  print(var.test(random, ge))
  print(t.test(random, ge, var.equal = TRUE))
  print(wilcox.test(random, ge))

  ggplot(data=stats, aes(label, duration)) +
    stat_summary(fun.y = mean, geom = "bar") +
    stat_summary(fun.data = mean_se, geom = "errorbar") +
    labs(
      y = "Duration per sample (s)"
    )
}

plot_ge_comparison_sd <- function(file) {
  stats <- read.csv(file = file, header = TRUE, stringsAsFactors = FALSE)
  stats$label <- factor(stats$label, levels = stats$label)

  ggplot(data=stats, aes(label, sd)) +
    geom_bar(stat="identity")
}

plot_ge_size_sampling <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot(
    data=stats,
    aes(
      x = factor(as.character(generations), levels = unique(generations)),
      y = factor(as.character(population), levels = unique(population)),
      size = average,
      color = duration / 20 / 60,
      label = sprintf("%0.2f", round(average, digits = 2))
    )
  ) +
    geom_point(shape = 15) +
    labs(
      x = "Generations",
      y = "Population size",
      color = "Mean duration (m)",
      size = "Mean score"
    ) +
    guides(size = FALSE) +
    scale_size_continuous(range = c(0.34, 1) * 20) +
    scale_color_gradientn(colors = c("#00AA00", "#FFFF00", "#FF0000", "#AA0000")) +
    # scale_color_gradientn(colors = c("#FF0000", "#FFFF00", "#00FF00", "#00AA00"))
    geom_text(size = 5, color = "black")
}

plot_ge_tournament_sampling <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot(
    data=stats,
    aes(
      x = factor(as.character(size), levels = unique(size)),
      y = average,
      fill = duration / 40 / 60,
    )
  ) +
    ylim(0, 1) +
    geom_bar(stat="identity") +
    geom_errorbar(aes(ymin = average - sqrt(variance) / sqrt(40), ymax = average + sqrt(variance) / sqrt(40)), width=0.8, color="red3") +
    labs(
      x = "Tournament size",
      y = "Mean score",
      fill = "Mean duration (m)"
    ) +
    scale_fill_gradientn(colors = c("#49f770", "#ffed5e", "#ff3155"))
}

plot_ge_recombination_sampling <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot(
    data=stats,
    aes(
      x = factor(as.character(crossover_rate), levels = unique(crossover_rate)),
      y = factor(as.character(mutation_rate), levels = unique(mutation_rate)),
      size = average,
      color = duration / 20 / 60,
      label = sprintf("%0.2f", round(average, digits = 2))
    )
  ) +
    geom_point(shape = 15) +
    labs(
      x = "Crossover rate",
      y = "Mutation rate",
      color = "Mean duration (m)",
      size = "Mean score"
    ) +
    guides(size = FALSE) +
    scale_size_continuous(range = c(0.66, 1) * 20) +
    scale_color_gradientn(colors = c("#00AA00", "#FFFF00", "#FF0000", "#AA0000")) +
    geom_text(size = 5, color = "black")
}

plot_ge_recombination_sampling_variance <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot(
    data=stats,
    aes(
      x = factor(as.character(crossover_rate), levels = unique(crossover_rate)),
      y = factor(as.character(mutation_rate), levels = unique(mutation_rate)),
      size = sqrt(variance),
      label = sprintf("%0.2f", round(sqrt(variance), digits = 2))
    )
  ) +
    geom_point(shape = 15) +
    labs(
      x = "Crossover rate",
      y = "Mutation rate",
      size = "SD"
    ) +
    guides(size = FALSE) +
    scale_size_continuous(range = c(0.04, 1) * 20) +
    geom_text(size = 4, color = "red3")
}

plot_ge_fitness_distribution <- function(file) {
  data <- read.csv(file = file, header = TRUE)

  ggplot(
    data = data,
    aes(
      x = fitness
    )
  ) +
    xlim(-0.05, 1.05) +
    geom_histogram(aes(y = ..density..), breaks = seq(0, 1, 0.05), colour="black", fill="white",) +
    geom_density(alpha = 0.2, fill = "#FF6666") +
    geom_vline(aes(xintercept = mean(fitness, na.rm = T)), color = "red", linetype = "dashed", size = 0.5)
}

plot_ge_scores_distribution <- function(file) {
  data <- read.csv(file = file, header = TRUE)

  ggplot(
    data = data,
    aes(
      x = best
    )
  ) +
    xlim(-0.05, 1.05) +
    geom_histogram(aes(y = ..density..), breaks = seq(0, 1, 0.01), colour="black", fill="white") +
    geom_density(alpha = 0.2, fill = "#FF6666") +
    geom_vline(aes(xintercept = mean(best, na.rm = T)), color = "red", linetype = "dashed", size = 1)
}
