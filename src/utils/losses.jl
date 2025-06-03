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
    y = ds_k(hm.targets)
    ŷ, αst = HM(ds, ps, st)
    _, st = αst
    # loss = mean((y[no_nan] .- ŷ[no_nan]).^2)
    l1 = mean((y(:soc) .- ŷ.SOCconc).^2)
    return loss
end