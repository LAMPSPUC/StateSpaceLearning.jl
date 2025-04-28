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