# Inspiration / Motivation

I had previous wrote a feed-forward neural network
(https://www.christopheroei.com/b/ad8197e069984d094bb771e8f73545287087219a6da1509941480e05cf0b4e96.jl)
to test out the Universal Approximation Theorem. It did pretty well (see
https://www.christopheroei.com/b/c8927dd3c5490f5dd525e6c2bc99b35b4d7fb073a1c15572b634c08982d96f7f.png).

After reading https://cdn.openai.com/papers/whisper.pdf
I thought that by training a single network on related timeseries, the performance on a
particular timeseries might be enhanced -- the network might gain a deeper understanding
of the economy at a particular time, and thereby give better predictions.

# Usage

* Download CSV files from https://fred.stlouisfed.org/ and put them into an empty directory.
* Edit `time_regained.jl` and set the `datadir` variable to point to your data directory.
* You'll also need to edit the script to specify whether you expect the time series to be linear or exponential. For example, the Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCNS, available from https://fred.stlouisfed.org/series/CPIAUCNS) should be exponential:
```Julia
seriestype["CPIAUCNS"] = "exponential"
```
* You'll need to install some dependencies. I used Julia version 1.8.3 (the latest as of 2022-11-25).
```Julia
(@v1.8) pkg> add CSV, DataFrames, Flux, Stardates, TimeZones
```
* Load the script:
```Julia
julia> include("fullpathnameforthescript/time_regained.jl")
```
* Wait for the training to complete.
* Now you can make predictions. This is you you ask the model to predict the value of CPIAUCNS halfway through 2022:
```Julia
julia> predict("CPIAUCNS", 2022.5)
300.98902830438476
```
