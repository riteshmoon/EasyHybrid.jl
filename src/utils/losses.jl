export lossfn
export LogLoss

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

struct LogLoss end

"""
    lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
"""
function lossfn(HM::RespirationRbQ10, ds_p, (ds_t, ds_t_nan), ps, st, ::LogLoss)
    ŷ, _ = HM(ds_p, ps, st)
    y = ds_t(HM.targets)
    y_nan = ds_t_nan(HM.targets)
    
    name_keys =  axiskeys(y, 1)
    losses = [mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)])) for k in name_keys]

    return NamedTuple{Tuple(name_keys)}(losses)
end