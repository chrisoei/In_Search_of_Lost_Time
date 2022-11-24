import CSV
import Flux
import Stardates
import TimeZones

datadir = "Z:\\data\\fred.stlouisfed.org"

datafiles = readdir(datadir)
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


