export lossfn
export LoggingLoss

"""
    lossfn(lhm::LinearHM, ds, (y, no_nan), ps, st)
"""
function lossfn(lhm::LinearHM, ds, (y, no_nan), ps, st)
    ŷ, αst = lhm(ds, ps, st)
    _, st = αst
    loss = mean((y[no_nan] .- ŷ[no_nan]).^2)
    return loss
end


"""
    lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st)
"""
function lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st)
    ŷ, _ = HM(ds_p, ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)

    loss = 0.0
    for k in axiskeys(y, 1)
        loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
    end
    return loss
end

"""
    lossfn(HM::RespirationRbQ10, ds, y, ps, st)
"""
function lossfn(HM::BulkDensitySOC, ds_p, (ds_t, ds_t_nan), ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)
    ŷ, _ = HM(ds_p, ps, st)

    loss = 0.0
    for k in axiskeys(y, 1)
        loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
    end    
    return loss
end

Base.@kwdef struct LoggingLoss{T<:Function}
    fn::T = sum
end

"""
    lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
"""
function lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st, logging::LoggingLoss)
    ŷ, _ = HM(ds_p, ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)
    
    name_keys =  axiskeys(y, 1)
    losses = [mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)])) for k in name_keys]
    loss = logging.fn(losses)
    return NamedTuple{(name_keys..., :sum)}([losses..., loss])
end

"""
    lossfn(HM::BulkDensitySOC ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
"""
function lossfn(HM::BulkDensitySOC, ds_p, (ds_t, ds_t_nan), ps, st, logging::LoggingLoss)
    ŷ, _ = HM(ds_p, ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)
    
    name_keys =  axiskeys(y, 1)
    losses = [mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)])) for k in name_keys]
    loss = logging.fn(losses)
    return NamedTuple{(name_keys..., :sum)}([losses..., loss])
end

"""
    lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
"""
function lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st, logging::LoggingLoss)
    ŷ, _ = HM(ds_p, ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)

    name_keys =  axiskeys(y, 1)
    losses = [mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)])) for k in name_keys]
    loss = logging.fn(losses)
    return NamedTuple{(name_keys..., :sum)}([losses..., loss])
end

"""
    lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st)
"""
function lossfn(HM::Rs_components, ds_p, (ds_t, ds_t_nan), ps, st)
    ŷ, _ = HM(ds_p, ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)

    loss = 0.0
    for k in axiskeys(y, 1)
        loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
    end    
    return loss
end