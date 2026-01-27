@testset "Function: fit!" begin
    y1 = rand(100)
    y2 = rand(100)
    y2[10:20] .= NaN

    model1 = StateSpaceLearning.StructuralModel(y1)
    StateSpaceLearning.fit!(model1)

    @test length(model1.output.ε) == 100
    @test length(model1.output.fitted) == 100
    @test length(model1.output.coefs) == 375
    @test length(model1.output.valid_indexes) == 100
    @test length(model1.output.residuals_variances) == 4
    @test length(keys(model1.output.components)) == 10
    @test length(keys(model1.output.decomposition)) == 3

    model2 = StateSpaceLearning.StructuralModel(y2)
    StateSpaceLearning.fit!(model2)

    @test length(model2.output.ε) == 100
    @test length(model2.output.fitted) == 100
    @test length(model2.output.coefs) == 375
    @test length(model2.output.valid_indexes) == 89
    @test length(model2.output.residuals_variances) == 4
    @test length(keys(model2.output.components)) == 10
    @test length(keys(model2.output.decomposition)) == 3
end

@testset "Function: fit_split!" begin
    Random.seed!(123)
    y = rand(100)
    
    # Test basic functionality with default parameters
    model1 = StateSpaceLearning.StructuralModel(y)
    StateSpaceLearning.fit_split!(model1; H=10)
    
    @test model1.output !== nothing
    @test length(model1.output.ε) == 100
    @test length(model1.output.fitted) == 100
    @test length(model1.output.coefs) > 0
    @test model1.output.coefs !== nothing
    
    # Test with custom α_set
    model2 = StateSpaceLearning.StructuralModel(y)
    StateSpaceLearning.fit_split!(model2; H=5, α_set=[0.1, 0.5, 1.0])
    
    @test model2.output !== nothing
    @test length(model2.output.ε) == 100
    @test length(model2.output.fitted) == 100
    
    # Test with different H values
    model3 = StateSpaceLearning.StructuralModel(y)
    StateSpaceLearning.fit_split!(model3; H=15, information_criteria="aic")
    
    @test model3.output !== nothing
    @test length(model3.output.ε) == 100
    
    # Test with different information criteria
    model4 = StateSpaceLearning.StructuralModel(y)
    StateSpaceLearning.fit_split!(model4; H=8, information_criteria="bic")
    
    @test model4.output !== nothing
    @test length(model4.output.ε) == 100
    
    # Test that it throws an error with exogenous variables
    exog = rand(100, 2)
    model5 = StateSpaceLearning.StructuralModel(y; exog=exog)
    @test_throws AssertionError StateSpaceLearning.fit_split!(model5; H=10)
    
    # Test with model that has no exogenous variables but different structure
    model6 = StateSpaceLearning.StructuralModel(
        y;
        level="stochastic",
        slope="none",
        seasonal="stochastic",
        freq_seasonal=12,
        outlier=false
    )
    StateSpaceLearning.fit_split!(model6; H=12)
    
    @test model6.output !== nothing
    @test length(model6.output.ε) == 100
    
    # Test that the selected α is from the α_set
    model7 = StateSpaceLearning.StructuralModel(y)
    α_set_custom = [0.2, 0.4, 0.6, 0.8]
    StateSpaceLearning.fit_split!(model7; H=10, α_set=α_set_custom)
    
    @test model7.output !== nothing
    # The model should be fitted with one of the α values from the set
    # We can't directly check which α was used, but we can verify the model is fitted
    @test length(model7.output.coefs) > 0
end
