export train

"""
    train(hybridModel, data; nepochs=200, batchsize=10, opt=Adam(0.01))
"""
function train(hybridModel, data, save_ps; nepochs=200, batchsize=10, opt=Adam(0.01),
    file_name=nothing,
        # metrics =( :mse, :nse) # TODO: include a list of metrics
        )
    # all the KeyedArray thing!

    # ? split training and validation data
    (x_train, y_train), (x_val, y_val) = splitobs(data; at=0.8, shuffle=false)
    train_loader = DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true);
    # ? setup model
    ps, st = LuxCore.setup(Random.default_rng(), hybridModel)
    opt_state = Optimisers.setup(opt, ps)

    # ? initial losses
    is_no_nan_t = .!isnan.(y_train)
    is_no_nan_v = .!isnan.(y_val)
    l_init_train = lossfn(hybridModel, x_train, (y_train, is_no_nan_t), ps, st, LoggingLoss())
    l_init_val = lossfn(hybridModel, x_val, (y_val, is_no_nan_v), ps, st, LoggingLoss())

    train_history = [l_init_train]
    val_history = [l_init_val]
    # track physical parameters
    ps_values_init = [copy(getproperty(ps, e)[1]) for e in save_ps]
    ps_init = NamedTuple{save_ps}(ps_values_init)
    ps_history = [ps_init]
    
    file_name = resolve_path(file_name)
    save_ps_st(file_name, hybridModel, ps, st, save_ps)
    save_train_val_loss!(file_name,l_init_train, "training_loss", 0)
    save_train_val_loss!(file_name,l_init_val, "validation_loss", 0)

    prog = Progress(nepochs, desc="Training loss")
    for epoch in 1:nepochs
        for (x, y) in train_loader
            # ? check NaN indices before going forward, and pass filtered `x, y`.
            is_no_nan = .!isnan.(y)
            if length(is_no_nan)>0 # ! be careful here, multivariate needs fine tuning
                grads = Zygote.gradient((ps) -> lossfn(hybridModel, x, (y, is_no_nan), ps, st), ps)[1] #TODO do we have to do losses then one with logging, one without
                Optimisers.update!(opt_state, ps, grads)
            end
        end
        save_ps_st!(file_name, hybridModel, ps, st, save_ps, epoch)

        ps_values = [copy(getproperty(ps, e)[1]) for e in save_ps]
        tmp_e = NamedTuple{save_ps}(ps_values)
        push!(ps_history, tmp_e)

        l_train = lossfn(hybridModel, x_train,  (y_train, is_no_nan_t), ps, st, LoggingLoss())
        l_val = lossfn(hybridModel, x_val, (y_val, is_no_nan_v), ps, st, LoggingLoss())
        # out_metrics = [m() for m in metrics] # TODO: include a list of metrics!
        save_train_val_loss!(file_name, l_train, "training_loss", epoch)
        save_train_val_loss!(file_name, l_val, "validation_loss", epoch)
        
        push!(train_history, l_train)
        push!(val_history, l_val)

        _headers, paddings = header_and_paddings(l_init_train)

        next!(prog; showvalues = [
            ("epoch ", epoch),
            ("targets ", join(_headers, "  ")),
            (styled"{red:training-start }", styled_values(l_init_train; paddings)),
            (styled"{bright_red:current }", styled_values(l_train; color=:bright_red, paddings)),
            (styled"{cyan:validation-start }", styled_values(l_init_val; paddings)),
            (styled"{bright_cyan:current }", styled_values(l_val; color=:bright_cyan, paddings)),
            ]
            )
            # TODO: log metrics
    end

    train_history = WrappedTuples(train_history)
    val_history = WrappedTuples(val_history)
    ps_history = WrappedTuples(ps_history)

    # ? save final evaluation or best at best validation value

    ŷ_train, αst_train = hybridModel(x_train, ps, st)
    ŷ_val, αst_val = hybridModel(x_val, ps, st)
    save_predictions!(file_name, ŷ_train, αst_train, "training")
    save_predictions!(file_name, ŷ_val, αst_val, "validation")

    # training
    target_names = hybridModel.targets
    save_observations!(file_name, target_names, y_train, "training")
    save_observations!(file_name, target_names, y_val, "validation")
    # save split obs (targets)

    # ? this could be saved to disk if needed for big sizes.
    train_obs = toDataFrame(y_train)
    train_hats = toDataFrame(ŷ_train, target_names)
    train_obs_pred = hcat(train_obs, train_hats)
    # validation
    val_obs = toDataFrame(y_val)
    val_hats = toDataFrame(ŷ_val, target_names)
    val_obs_pred = hcat(val_obs, val_hats)
    # ? diffs, additional predictions without observational counterparts!
    # TODO: better!
    set_diff = setdiff(keys(ŷ_train), target_names)
    train_diffs = !isempty(set_diff) ? NamedTuple{Tuple(set_diff)}([getproperty(ŷ_train, e) for e in set_diff]) : nothing 
    val_diffs = !isempty(set_diff) ? NamedTuple{Tuple(set_diff)}([getproperty(ŷ_val, e) for e in set_diff]) : nothing

    # TODO: save/output metrics

    return (; train_history, val_history, ps_history, train_obs_pred, val_obs_pred, train_diffs, val_diffs, αst_train, αst_val, ps, st)
end

function styled_values(nt; digits=5, color=nothing, paddings=nothing)
    formatted = [
        begin
            value_str = @sprintf("%.*f", digits, v)
            padded = isnothing(paddings) ? value_str : rpad(value_str, paddings[i])
            isnothing(color) ? padded  : styled"{$color:$padded}"
        end
        for (i,v) in enumerate(values(nt))
    ]
    return join(formatted, "  ")
end

function header_and_paddings(nt; digits=5)
    min_val_width = digits + 2  # 1 for "0", 1 for ".", rest for digits
    paddings = map(k -> max(length(string(k)), min_val_width), keys(nt))
    headers = [rpad(string(k), w) for (k, w) in zip(keys(nt), paddings)]
    return headers, paddings
end