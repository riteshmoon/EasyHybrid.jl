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
    lossfn(HM::RespirationRbQ10, ds, y, ps, st)
"""
function lossfn(HM::RespirationRbQ10, ds, (y, no_nan), ps, st)
    ŷ, αst = HM(ds, ps, st)
    _, st = αst
    loss = mean((y[no_nan] .- ŷ[no_nan]).^2)
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