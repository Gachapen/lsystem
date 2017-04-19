stats <- read.csv(file="stats.csv", header=TRUE)

stats <- split(stats, stats$distribution) # Create list with each distribution
stats <- lapply(stats, subset, select = -distribution) # Remove distribution variable

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
	barplot(success_rate, ylim=c(0,3), ylab="% of something", xlab="distribution level")
}

score_changes <- function(stats) {
	mean_scores <- lapply(stats, lapply, mean, na.rm=TRUE)
	mean_scores <- matrix(unlist(mean_scores), ncol=3) # transform to matrix
	colnames(mean_scores) <- c(0, 1, 2)
	rownames(mean_scores) <- c("score", "balance", "branching", "closeness", "drop")

	# Negate closenss and drop scores as they are punishments, not rewards.
	mean_scores["closeness",] <- -mean_scores["closeness",]
	mean_scores["drop",] <- -mean_scores["drop",]

	colors <- rainbow(5)
	pch <- 21:25
	lty <- 1:5
	plot(0, 0, xlim=c(0, 2), ylim=c(-1.5, 1.0), ylab="score", xlab="distribution level", type="n")
	for (i in 1:5) {
		lines(c(0, 1, 2), mean_scores[i, ], type="o", col=colors[i], pch=pch[i], lty=lty[i])
	}
	legend(0, 1, rownames(mean_scores), col=colors, cex=0.8, pch=pch, lty=lty)
}
