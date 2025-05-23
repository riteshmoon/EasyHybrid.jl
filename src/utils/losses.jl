export lossfn

"""
    lossfn(lhm::LinearHM, ds, y, ps, st)
"""
function lossfn(lhm::LinearHM, ds, y, ps, st)
    ŷ, αst = lhm(ds, ps, st)
    _, st = αst
    loss = mean((y .- ŷ).^2)
    return loss
end