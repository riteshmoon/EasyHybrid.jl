function poplot()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `poplot` with no arguments!")
end

function poplot!()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `poplot!` with no arguments!")
end

function plot_pred_vs_obs!()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_pred_vs_obs!` with no arguments!")
end

function train_board()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_pred_vs_obs!` with no arguments!")
end


function plot_parameters!()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_parameters!` with no arguments!")
end

function plot_loss!()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_loss!` with no arguments!")
end

function plot_parameters()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_parameters` with no arguments!")
end

function plot_training_summary()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_training_summary` with no arguments!")
end
function update_plotting_observables()
    @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_training_summary` with no arguments!")
end

function plot_loss end
function to_obs end
function to_point2f end
function update_plotting_observables end


"""
    initialize_plotting_observables(ext, hybridModel, x_train, y_train, x_val, y_val, l_init_train, l_init_val, training_loss, agg, monitor_names, ps, st)

Initialize plotting observables for training visualization if the Makie extension is loaded.
"""
function initialize_plotting_observables(hybridModel, x_train, y_train, x_val, y_val, l_init_train, l_init_val, training_loss, agg, monitor_names, ps, st)
    target_names = hybridModel.targets

    # Initialize loss history observables
    l_value = getproperty(getproperty(l_init_train, training_loss), Symbol("$agg"))
    p = EasyHybrid.to_point2f(0, l_value)
    train_h_obs = EasyHybrid.to_obs([p])
    
    l_value_val = getproperty(getproperty(l_init_val, training_loss), Symbol("$agg"))
    p_val = EasyHybrid.to_point2f(0, l_value_val)
    val_h_obs = EasyHybrid.to_obs([p_val])

    # Initial predictions for scatter plot obs versus model
    ŷ_train = hybridModel(x_train, ps, LuxCore.testmode(st))[1]
    ŷ_val = hybridModel(x_val, ps, LuxCore.testmode(st))[1]

    # build NamedTuples of Observables for preds and obs
    train_preds = to_obs_tuple(ŷ_train, target_names)
    val_preds   = to_obs_tuple(ŷ_val, target_names)
    train_obs   = to_obs_tuple(y_train, target_names)
    val_obs     = to_obs_tuple(y_val, target_names)

    # --- monitored parameters/state as Observables ---
    train_monitor = monitor_to_obs(ŷ_train, monitor_names)
    val_monitor =  monitor_to_obs(ŷ_val, monitor_names)

    return (; train_h_obs, val_h_obs, train_preds, train_obs, val_preds, val_obs, train_monitor, val_monitor)
end

function to_obs_tuple(y, target_names)
    return (; (t => EasyHybrid.to_obs(vec(getfield(y, t))) for t in target_names)...)
end

function to_obs_tuple(y::KeyedArray, target_names)
    return (; (t => EasyHybrid.to_obs(y(t)) for t in target_names)...)
end

function monitor_to_obs(ŷ, monitor_names; cuts = (0.25, 0.5, 0.75))
    return (; (
        m => begin
            v = vec(getfield(ŷ, m))
            if length(v) > 1
                qx_ = Symbol("q$(Int(q*100))")
                point_ = [EasyHybrid.to_point2f(0, quantile(v, q))]
                # return
                (; (qx_ => EasyHybrid.to_obs(point_) for q in cuts)...)
            else
                point_ = [EasyHybrid.to_point2f(0, v[1])]
                # return
                (; :scalar => EasyHybrid.to_obs(point_))
            end
        end for m in monitor_names)...)
end