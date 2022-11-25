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
seriestype["GASREGW"] = "exponential"
seriestype["GDPA"] = "linear"
seriestype["MEDDAYONMAR29820"] = "linear"
seriestype["MORTGAGE30US"] = "linear"
seriestype["UNRATE"] = "linear"
seriestype["WILL5000IND"] = "exponential"
seriesnames = map(x -> replace(x, r"\.csv$" => ""), datafiles)
@assert(Set(map(x -> seriestype[x], seriesnames)) == Set(["exponential", "linear"]))
seriests = Dict{String, Vector{Float64}}()
seriesvs = Dict{String, Vector{Float64}}()
min_time = nothing
max_time = nothing

function date2stardate(d)::Float64
    return Stardates.Stardate(d, 16, 0, 0,
    TimeZones.tz"America/New_York").sd
end

for s in seriesnames
    global min_time, max_time
    println(s)
    tmp = CSV.File(joinpath(datadir, s * ".csv"),
                   header = ["DATE", "VALUE"],
                   skipto = 2,
                   types = Dict("DATE" => Dates.Date, "VALUE" => Float64),
                   missingstring=["."])
    tmp = filter(x->typeof(x[:VALUE])<:Number, tmp)
    seriesdata[s] = tmp

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
    end
end

inputrows = []
outputrows = []
for i in 1:nseries
    s = seriesnames[i]
    println(s)
    for t in seriests[s]
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
        prepend!(outputrows, [map(x -> [x], seriesvs[s])])
    end
end

@info "min_time = $min_time"
@info "max_time = $max_time"

# vim: set et ff=unix ft=julia nocp sts=4 sw=4 ts=4:
