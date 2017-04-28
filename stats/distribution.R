require(ggplot2)
require(gridExtra)

distribution <- read.csv(file = "distribution.csv", header = TRUE)

productions <- function(distribution) {
  prods <- distribution[distribution$rule == "productions",]
  ggplot(prods, aes(alternative + 1, weight)) +
    geom_col() +
    xlab("num productions") +
    ggtitle("production count distribution")
}

operation <- function(distribution) {
  operations <- distribution[distribution$rule == "operation",]
  ggplot(operations, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="operation")) +
    scale_fill_discrete(labels=c("+", "-", "^", "&", ">", "<")) +
    ggtitle("operation distribution")
}

strlen <- function(distribution) {
  strlens <- distribution[distribution$rule == "string" & distribution$choice == 0,]
  ggplot(strlens, aes(depth, weight, fill=factor(alternative + 1))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="string length", ncol=2)) +
    ggtitle("string length distribution")
}

stack <- function(distribution) {
  stacks <- distribution[distribution$rule == "string" & distribution$choice == 1,]
  ggplot(stacks, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="symbol vs stack")) +
    scale_fill_discrete(labels=c("symbol", "stack")) +
    ggtitle("symbol/stack distribution")
}

symbol <- function(distribution) {
  symbols <- distribution[distribution$rule == "symbol",]
  ggplot(symbols, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="symbol type")) +
    scale_fill_discrete(labels=c("variable", "operation")) +
    ggtitle("symbol distribution")
}

variable <- function(distribution) {
  variables <- distribution[distribution$rule == "variable",]
  ggplot(variables, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="variable", ncol=2)) +
    scale_fill_discrete(labels=LETTERS) +
    ggtitle("variable distribution")
}

plot_all <- function(file) {
  distribution <- read.csv(file = file, header = TRUE)

  grid.arrange(
    productions(distribution),
    strlen(distribution),
    stack(distribution),
    symbol(distribution),
    variable(distribution),
    operation(distribution),
    nrow=3,
    ncol=2)
}

plot_directory <- function(dir) {
  dists <- list.files(dir, full.names = TRUE)
  for (dist in dists) {
    distribution <- read.csv(file = dist, header = TRUE)

    g <- arrangeGrob(
      productions(distribution),
      strlen(distribution),
      stack(distribution),
      symbol(distribution),
      variable(distribution),
      operation(distribution),
      nrow=3,
      ncol=2)

    png <- paste(dist, ".png", sep = "")
    ggsave(file = png, g)
  }
}
