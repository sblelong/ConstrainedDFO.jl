function obj_solar(x::Vector{Float64}, pb_id::Int; path_base::String = "/home/sblelong/.julia/dev/ConstrainedDFO/", solar_path::String = "/home/sblelong/dev/solar/", return_inequalities::Bool = false)
    # Write x in a .txt file
    x_path = joinpath(path_base, "x.txt")
    open(x_path, "w") do io
        println(io, join(x, " "))
    end

    # Run SOLAR
    solar_exec = joinpath(solar_path, "bin/solar")
    f = readchomp(ignorestatus(`$(solar_exec) $(pb_id) $(x_path)`))
    if return_inequalities
        return parse.(Float64, split(f))
    else
        return parse(Float64, split(f)[1])
    end
    rm(x_path)
    return f
end

export obj_solar
