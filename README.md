# Data

* https://fred.stlouisfed.org/series/CPIAUCNS
* ...

# Inspirations

I had previous wrote a feed-forward neural network
(https://www.christopheroei.com/b/ad8197e069984d094bb771e8f73545287087219a6da1509941480e05cf0b4e96.jl)
to test out the Universal Approximation Theorem.

After reading https://cdn.openai.com/papers/whisper.pdf
I thought that by training a single network on related timeseries, the performance on a
particular timeseries might be enhanced -- the network might gain a deeper understanding
of the economy at a particular time, and thereby give better predictions.

