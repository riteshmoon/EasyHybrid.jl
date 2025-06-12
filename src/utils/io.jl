function save_parameters(file_name, ps, st)
    jldsave(file_name; ps, st)
end

function save_parameters!(file_name, ps, st)
    jldopen("example.jld2", "w") do file
    file["bigdata"] = randn(5)
end
    jldsave(file_name; ps, st)
end

function resolve_path(file_name)
    file_name = isnothing(file_name) ? "trained_model.jld2" : file_name
    if !endswith(file_name, ".jld2")
        error("This needs to be a jld2 file, please include extension as in `file_name.jld2`")
    end
    file_name = if isabspath(file_name)
        return file_name
    else
        tmp_folder = mkpath(joinpath(@__DIR__, "output"))
        return joinpath(tmp_folder, file_name)
    end
    return file_name
end