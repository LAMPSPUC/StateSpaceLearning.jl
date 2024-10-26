@testset "Function: fit_model" begin
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
    @test length(keys(model1.output.components)) == 9

    model2 = StateSpaceLearning.StructuralModel(y2)
    StateSpaceLearning.fit!(model2)

    @test length(model2.output.ε) == 100
    @test length(model2.output.fitted) == 100
    @test length(model2.output.coefs) == 375
    @test length(model2.output.valid_indexes) == 89
    @test length(model2.output.residuals_variances) == 4
    @test length(keys(model2.output.components)) == 9
end

@testset "Function: forecast" begin
    y1 = rand(100)
    y2 = rand(100)
    y2[10:20] .= NaN

    model1 = StateSpaceLearning.StructuralModel(y1)
    StateSpaceLearning.fit!(model1)
    @test length(StateSpaceLearning.forecast(model1, 10)) == 10

    model2 = StateSpaceLearning.StructuralModel(y2; Exogenous_X=rand(100, 3))
    StateSpaceLearning.fit!(model2)
    @test length(StateSpaceLearning.forecast(model2, 10; Exogenous_Forecast=rand(10, 3))) ==
          10

    @test_throws AssertionError StateSpaceLearning.forecast(model1, 10;
                                                            Exogenous_Forecast=rand(5, 3))
    @test_throws AssertionError StateSpaceLearning.forecast(model2, 10)
    @test_throws AssertionError StateSpaceLearning.forecast(model2, 10;
                                                            Exogenous_Forecast=rand(5, 3))

    y3 = [4.718, 4.77, 4.882, 4.859, 4.795, 4.905, 4.997, 4.997, 4.912, 4.779, 4.644, 4.77,
          4.744, 4.836, 4.948, 4.905, 4.828, 5.003, 5.135, 5.135, 5.062, 4.89, 4.736, 4.941,
          4.976, 5.01, 5.181, 5.093, 5.147, 5.181, 5.293, 5.293, 5.214, 5.087, 4.983, 5.111,
          5.141, 5.192, 5.262, 5.198, 5.209, 5.384, 5.438, 5.488, 5.342, 5.252, 5.147,
          5.267, 5.278, 5.278, 5.463, 5.459, 5.433, 5.493, 5.575, 5.605, 5.468, 5.351,
          5.192, 5.303, 5.318, 5.236, 5.459, 5.424, 5.455, 5.575, 5.71, 5.68, 5.556, 5.433,
          5.313, 5.433, 5.488, 5.451, 5.587, 5.594, 5.598, 5.752, 5.897, 5.849, 5.743,
          5.613, 5.468, 5.627, 5.648, 5.624, 5.758, 5.746, 5.762, 5.924, 6.023, 6.003,
          5.872, 5.723, 5.602, 5.723, 5.752, 5.707, 5.874, 5.852, 5.872, 6.045, 6.142,
          6.146, 6.001, 5.849, 5.72, 5.817, 5.828, 5.762, 5.891, 5.852, 5.894, 6.075, 6.196,
          6.224, 6.001, 5.883, 5.736, 5.82, 5.886, 5.834, 6.006, 5.981, 6.04, 6.156, 6.306,
          6.326, 6.137, 6.008, 5.891, 6.003, 6.033, 5.968, 6.037, 6.133, 6.156, 6.282,
          6.432, 6.406, 6.23, 6.133, 5.966, 6.068]
    model3 = StateSpaceLearning.StructuralModel(y3)
    StateSpaceLearning.fit!(model3)
    forecast3 = trunc.(StateSpaceLearning.forecast(model3, 18); digits=3)
    @assert forecast3 ==
            [6.11, 6.082, 6.221, 6.19, 6.197, 6.328, 6.447, 6.44, 6.285, 6.163, 6.026,
             6.142, 6.166, 6.138, 6.278, 6.246, 6.253, 6.384]
end

@testset "Function: simulate" begin
    y1 = rand(100)
    y2 = rand(100)
    y2[10:20] .= NaN

    model1 = StateSpaceLearning.StructuralModel(y1)
    StateSpaceLearning.fit!(model1)
    @test size(StateSpaceLearning.simulate(model1, 10, 100)) == (10, 100)

    model2 = StateSpaceLearning.StructuralModel(y2; Exogenous_X=rand(100, 3))
    StateSpaceLearning.fit!(model2)
    @test size(StateSpaceLearning.simulate(model2, 10, 100; Exogenous_Forecast=rand(10, 3))) ==
          (10, 100)
end
