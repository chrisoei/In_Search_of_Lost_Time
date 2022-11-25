import CSV
import DataFrames
import Dates
import Flux
import Stardates
import TimeZones

datadir = "Z:\\data\\fred.stlouisfed.org"

datafiles = readdir(datadir)
nseries = length(datafiles)
seriestype = Dict{String, String}()
seriestype["ACTLISCOU29820"] = "exponential"
seriestype["CPIAUCNS"] = "exponential"
seriestype["FEDFUNDS"] = "linear"
seriestype["GASREGW"] = "exponential"
seriestype["GDPA"] = "exponential"
seriestype["LVXRNSA"] = "exponential"
seriestype["MEDDAYONMAR29820"] = "linear"
seriestype["MORTGAGE30US"] = "linear"
seriestype["MSACSR"] = "linear"
seriestype["SFXRNSA"] = "exponential"
seriestype["UNRATE"] = "linear"
seriestype["WILL5000IND"] = "exponential"
seriesnames = map(x -> replace(x, r"\.csv$" => ""), datafiles)
@assert(Set(map(x -> seriestype[x], seriesnames)) == Set(["exponential", "linear"]))
seriests = Dict{String, Vector{Float64}}()
seriesvs = Dict{String, Vector{Float64}}()
seriesminv = Dict{String, Float64}()
seriesmaxv = Dict{String, Float64}()
seriesindex = Dict{String, Int64}()
min_time = nothing
max_time = nothing

function date2stardate(d)::Float64
    return Stardates.Stardate(d, 16, 0, 0,
    TimeZones.tz"America/New_York").sd
end

@info "Preprocessing: first pass"
for i in 1:nseries
    s = seriesnames[i]
    seriesindex[s] = i
    global min_time, max_time
    println(s)
    tmp = CSV.File(joinpath(datadir, s * ".csv"),
                   header = ["DATE", "VALUE"],
                   skipto = 2,
                   types = Dict("DATE" => Dates.Date, "VALUE" => Float64),
                   missingstring=["."])
    tmp = filter(x->typeof(x[:VALUE])<:Number, tmp)

    ltmp = length(tmp)
    @info "length(tmp) = $ltmp"
    if length(tmp) > 0
        stardates = map(x->date2stardate(x[:DATE]), tmp)
        if min_time == nothing
            min_time = min(stardates...)
        else
            min_time = min(min(stardates...), min_time)
        end
        if max_time == nothing
            max_time = max(stardates...)
        else
            max_time = max(max(stardates...), max_time)
        end
        vals = map(x->x[:VALUE], tmp)
        if seriestype[s] == "exponential"
            vals = log.(vals)
        end
        seriests[s] = stardates
        seriesvs[s] = vals
        seriesminv[s] = min(vals...)
        seriesmaxv[s] = max(vals...)
    end
end

@info "Preprocessing: second pass"
inputrows = []
outputrows = []
for i in 1:nseries
    s = seriesnames[i]
    @info s
    for j in 1:length(seriests[s])
        t = seriests[s][j]
        global inputrows
        inputrow = zeros(nseries + 2)
        # Rescale time so that it falls between -1 and 1
        t1 = 2 * (t - min_time) / (max_time - min_time) - 1
        inputrow[1] = t1
        # Also include the fractional year. The neural network
        # could figure this out on its own, but since many timeseries
        # have a seasonal component, I thought I'd toss this in to
        # make things easier for it.
        inputrow[2] = 2 * (t - floor(t)) - 1
        # Tell the network which series this is.
        inputrow[i + 2] = 1.0
        prepend!(inputrows, [inputrow])
        # Rescale the output value so that it falls between -1 and 1
        v1 = 2 * (seriesvs[s][j] - seriesminv[s]) / (seriesmaxv[s] - seriesminv[s]) - 1
        prepend!(outputrows, [[v1]])
    end
end

@info "min_time = $min_time"
@info "max_time = $max_time"

data1 = Flux.Data.DataLoader(
    (inputrows, outputrows),
    batchsize = 8,
    shuffle = true)
@info "Creating model"
model = Flux.Chain(
    Flux.Dense(nseries + 2, 300, Flux.relu),
    Flux.Dense(300, 300, Flux.relu),
    Flux.Dense(300, 300, Flux.relu),
    Flux.Dense(300, 300, Flux.relu),
    Flux.Dense(300, 1, Flux.identity))

@info "Extracting parameters"
p = Flux.params(model)
@info "Defining loss function"
loss(x, y) = Flux.mse(model(x), y)
@info "Defining optimization"
learnrate = 0.01
opt = Flux.Descent(learnrate)
data0 = collect(zip(inputrows, outputrows))
currentloss = sum(map(x -> loss(x...), data0))
@info "currentloss:", currentloss, "learnrate:", learnrate

function predict(s::String, t::Float64)::Float64
    is1 = seriesindex[s]
    t1 = 2 * (t - min_time) / (max_time - min_time) - 1
    t2 = 2 * (t - floor(t)) - 1
    inputrow = zeros(nseries + 2)
    inputrow[1] = t1
    inputrow[2] = t2
    inputrow[is1 + 2] = 1.0
    v1 = model(inputrow)[1]
    v2 = (v1 + 1) * (seriesmaxv[s] - seriesminv[s]) / 2 + seriesminv[s]
    if seriestype[s] == "exponential"
        v2 = exp(v2)
    end
    return v2
end

for i1 in 1:100
    @info "Training iteration $i1"
    global oldloss = currentloss
    for d1 in data1
        l1 = length(d1[1])
        #@info "Sub-iteration length: $l1"
        batchx = d1[1]
        batchy = d1[2]
        gs = Flux.gradient(p) do
            s0 = sum(map(x->loss(x...), zip(batchx, batchy)))
            return s0
        end
        Flux.Optimise.update!(opt, p, gs)
        #s1 = sum(map(x->loss(x...), zip(batchx, batchy)))
        #@info "Sub-iteration loss: $s1"
        #@assert(!isnan(s1))
    end
    global currentloss = sum(map(x -> loss(x...), data0))
    if currentloss > oldloss
        global learnrate *= 0.9
        if learnrate < 1.0e-6
            break
        end
        global opt = Flux.Descent(learnrate)
    end
    @info "currentloss:", currentloss, "learnrate:", learnrate
    @assert(!isnan(currentloss))
end

# vim: set et ff=unix ft=julia nocp sts=4 sw=4 ts=4:
