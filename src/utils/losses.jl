export lossfn

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
function lossfn(HM::BulkDensitySOC, ds, ps, st)
    y = ds(HM.targets)
    ŷ, αst = HM(ds, ps, st)
    _, st = αst
    # loss = mean((y[no_nan] .- ŷ[no_nan]).^2)
    loss = 0.0
    for key in keys(ŷ)
        loss += mean((y(key) .- ŷ[key]).^2)
    end
    
    return loss
end