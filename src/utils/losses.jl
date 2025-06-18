export lossfn
export LoggingLoss

"""
    LoggingLoss

A structure to define a logging loss function for hybrid models.
It allows for multiple loss types and an aggregation function to be specified.

# Arguments:
- `loss_types::Vector{Symbol}`: A vector of loss types to compute, e.g., `[:mse, :mae]`.
- `training_loss::Symbol`: The loss type to use during training, e.g., `:mse`.
- `agg::Function`: A function to aggregate the losses, e.g., `sum` or `mean`.
- `train_mode::Bool`: A flag indicating whether the model is in training mode. If `true`, it uses `training_loss`; otherwise, it uses `loss_types`.
"""
Base.@kwdef struct LoggingLoss{T<:Function}
    loss_types::Vector{Symbol} = [:mse]
    training_loss::Symbol = :mse
    agg::T = sum
    train_mode::Bool = true
end

"""
    lossfn(HM::LuxCore.AbstractLuxContainerLayer, x, (y_t, y_nan), ps, st, logging::LoggingLoss)

Arguments:
- `HM::LuxCore.AbstractLuxContainerLayer`: The hybrid model to compute the loss for.
- `x`: Input data for the model.
- `(y_t, y_nan)`: Tuple containing the target values and a mask for NaN values.
- `ps`: Parameters of the model.
- `st`: State of the model.
- `logging::LoggingLoss`: Logging configuration for the loss function.
"""
function lossfn(HM::LuxCore.AbstractLuxContainerLayer, x, (y_t, y_nan), ps, st, logging::LoggingLoss)
    targets = HM.targets
    ŷ, y, y_nan = get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    if logging.train_mode
        return compute_loss(ŷ, y, y_nan, targets, logging.training_loss, logging.agg)
    else
        return compute_loss(ŷ, y, y_nan, targets, logging.loss_types, logging.agg)
    end
end

function get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    ŷ, st = HM(x, ps, st)
    y = y_t(HM.targets)
    y_nan = y_nan(HM.targets)
    return (ŷ, y, y_nan)
end
function compute_loss(ŷ, y, y_nan, targets, training_loss::Symbol, agg::Function)
    losses = [loss_fn(ŷ[k], y(k), y_nan(k), Val(training_loss)) for k in targets]
    return agg(losses)
end
function compute_loss(ŷ, y, y_nan, targets, loss_types::Vector{Symbol}, agg::Function)
    out_loss_types = [
        begin
            losses = [loss_fn(ŷ[k], y(k), y_nan(k), Val(loss_type)) for k in targets]
            agg_loss = agg(losses)
            NamedTuple{(targets..., Symbol(agg))}([losses..., agg_loss])
        end
        for loss_type in loss_types]
    return NamedTuple{Tuple(loss_types)}([out_loss_types...])
end

# add as many loss functions as needed
function loss_fn(ŷ, y, y_nan, ::Val{:rmse})
    return sqrt(mean(abs2, (ŷ[y_nan] .- y[y_nan])))
end
function loss_fn(ŷ, y, y_nan, ::Val{:mse})
    return mean(abs2, (ŷ[y_nan] .- y[y_nan]))
end
function loss_fn(ŷ, y, y_nan, ::Val{:mae})
    return mean(abs, (ŷ[y_nan] .- y[y_nan]))
end
# person correlation coefficient
function loss_fn(ŷ, y, y_nan, ::Val{:pearson})
    return cor(ŷ[y_nan], y[y_nan])
end
function loss_fn(ŷ, y, y_nan, ::Val{:r2})
    r = cor(ŷ[y_nan], y[y_nan])
    return r*r
end

# legacy
# """
#     lossfn(lhm::LinearHM, ds, (y, no_nan), ps, st)
# """
# function lossfn(lhm::LinearHM, ds, (y, no_nan), ps, st)
#     ŷ, αst = lhm(ds, ps, st)
#     _, st = αst
#     loss = mean((y[no_nan] .- ŷ[no_nan]).^2)
#     return loss
# end


# """
#     lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st)
# """
# function lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st)
#     ŷ, _ = HM(ds_p, ps, st)
#     y = ds_t(HM.targets)
#     y_nan = ds_t_nan(HM.targets)

#     loss = 0.0
#     for k in axiskeys(y, 1)
#         loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
#     end
#     return loss
# end

# """
#     lossfn(HM::RespirationRbQ10, ds, y, ps, st)
# """
# function lossfn(HM::BulkDensitySOC, ds_p, (ds_t, ds_t_nan), ps, st)
#     y = ds_t(HM.targets)
#     y_nan = ds_t_nan(HM.targets)
#     ŷ, _ = HM(ds_p, ps, st)

#     loss = 0.0
#     for k in axiskeys(y, 1)
#         loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
#     end    
#     return loss
# end

# Base.@kwdef struct LoggingLoss{T<:Function}
#     fn::T = sum
# end

# """
#     lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
# """
# function lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st, logging::LoggingLoss)
#     ŷ, _ = HM(ds_p, ps, st)
#     y = ds_t(HM.targets)
#     y_nan = ds_t_nan(HM.targets)
    
#     name_keys =  axiskeys(y, 1)
#     losses = [mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)])) for k in name_keys]
#     loss = logging.fn(losses)
#     return NamedTuple{(name_keys..., :sum)}([losses..., loss])
# end

# """
#     lossfn(HM::BulkDensitySOC ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
# """
# function lossfn(HM::BulkDensitySOC, ds_p, (ds_t, ds_t_nan), ps, st, logging::LoggingLoss)
#     ŷ, _ = HM(ds_p, ps, st)
#     y = ds_t(HM.targets)
#     y_nan = ds_t_nan(HM.targets)
    
#     name_keys =  axiskeys(y, 1)
#     losses = [mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)])) for k in name_keys]
#     loss = logging.fn(losses)
#     return NamedTuple{(name_keys..., :sum)}([losses..., loss])
# end

# """
#     lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
# """
# function lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st, logging::LoggingLoss)
#     ŷ, _ = HM(ds_p, ps, st)
#     y = ds_t(HM.targets)
#     y_nan = ds_t_nan(HM.targets)

#     name_keys =  axiskeys(y, 1)
#     losses = [mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)])) for k in name_keys]
#     loss = logging.fn(losses)
#     return NamedTuple{(name_keys..., :sum)}([losses..., loss])
# end

# """
#     lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st)
# """
# function lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st)
#     ŷ, _ = HM(ds_p, ps, st)
#     y = ds_t(HM.targets)
#     y_nan = ds_t_nan(HM.targets)

#     loss = 0.0
#     for k in axiskeys(y, 1)
#         loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
#     end    
#     return loss
# end