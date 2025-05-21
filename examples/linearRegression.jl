using EasyHybrid

df = gen_linear_data(; seed=123)

## Instantiate model

hModel = LinearHybridModel([:x2, :x3], [:x1], 1, 5, b=[0.0f0])

# Fit the model with the non-stateful function fit_df! to the first half of the data set
# One does not need to put predictors explicitly, if they are explicit in the model

# res = fit_df!(hModel, df[1:500,:], [:obs], Flux.mse, n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=[:b], latents2record=[:pred => :obs, :a => "a_syn"], patience=300, stateful=false);


##
# test_df = df[501:1000, :]

### Make a direct evaluation
# fig2=evalfit(res, test_df)
