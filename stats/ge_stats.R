library(ggplot2)
library(gridExtra)

plot_ge <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot() +
    ylim(0, 1) +
    geom_point(data = stats, aes(iteration, avg), size = 1, color = "blue3") +
    geom_point(data = stats, aes(iteration, best), size = 1, color = "green3")
}
