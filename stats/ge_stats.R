library(ggplot2)
library(gridExtra)

plot_ge <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot() +
    ylim(0, 1) +
    geom_point(data = stats, aes(iteration, avg), size = 0.5, color = "blue3") +
    geom_point(data = stats, aes(iteration, best), size = 0.5, color = "green3")
}

plot_ge_comparison <- function(file) {
  stats <- read.csv(file = file, header = TRUE, stringsAsFactors = FALSE)
  stats$label <- factor(stats$label, levels = stats$label)

  ggplot(data=stats, aes(label, mean)) +
    ylim(0, 1) +
    geom_bar(stat="identity") +
    geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=0.8) +
    geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=0.8, color="red3")
}

plot_ge_comparison_sd <- function(file) {
  stats <- read.csv(file = file, header = TRUE, stringsAsFactors = FALSE)
  stats$label <- factor(stats$label, levels = stats$label)

  ggplot(data=stats, aes(label, sd)) +
    ylim(0, 0.1) +
    geom_bar(stat="identity")
}
