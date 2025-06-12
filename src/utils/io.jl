export  load_group
function save_ps_st(file_name, hm, ps, st, save_ps)
    hm_name = string(nameof(typeof(hm)))
    # split physical parameters
    tmp_e = if !isempty(save_ps)
        ps_values = [copy(getproperty(ps, e)[1]) for e in save_ps]
        NamedTuple{save_ps}(ps_values)
    end

    jldopen(file_name, "w") do file
        file["$hm_name/epoch_0"] = (ps, st)
        if !isempty(save_ps)
            file["physical_params/epoch_0"] = tmp_e
        end
    end
end

function save_ps_st!(file_name, hm, ps, st, save_ps, epoch)
    hm_name = string(nameof(typeof(hm)))
    # split physical parameters
    tmp_e = if !isempty(save_ps)
        ps_values = [copy(getproperty(ps, e)[1]) for e in save_ps]
        NamedTuple{save_ps}(ps_values)
    end

    jldopen(file_name, "a+") do file
        file["$hm_name/epoch_$epoch"] = (ps, st)
        if !isempty(save_ps)
            file["physical_params/epoch_$epoch"] = tmp_e
        end
    end
end

function load_group(file_name, group)
    group_tmp = JLD2.load(file_name, group)
    epoch_keys = collect(keys(group_tmp))
    sorted_keys = sort(epoch_keys, by = k -> parse(Int, split(k, "_")[end]))
    collected_data = [group_tmp[k] for k in sorted_keys]
    
    return collected_data, sorted_keys
end

function resolve_path(file_name)
    file_name = isnothing(file_name) ? "trained_model.jld2" : file_name
    if !endswith(file_name, ".jld2")
        error("This needs to be a jld2 file, please include the extension as in `file_name.jld2`")
    end
    file_name = if isabspath(file_name)
        return file_name
    else
        tmp_folder = mkpath(joinpath(dirname(Base.active_project()), "output_tmp"))
        return joinpath(tmp_folder, file_name)
    end
    return file_name
end