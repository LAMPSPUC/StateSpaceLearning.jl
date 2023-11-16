@testset "Function: fit_model" begin
    y1 = rand(100)
    y2 = rand(100); y2[10:20] .= NaN

    output1 = StateSpaceLearning.fit_model(y1)
    @test length(output1.ϵ) == 100
    @test length(output1.fitted) == 100
    @test size(output1.X, 1) == 100
    @test size(output1.X, 2) == length(output1.coefs)
    
    output2 = StateSpaceLearning.fit_model(y2)
    @test output2.valid_indexes == setdiff(1:100, 10:20)
    @test all(isnan.(output2.ϵ[10:20]))
    @test !all(isnan.(output2.fitted[10:20]))

    output3 = StateSpaceLearning.fit_model(y1; stabilize_ζ = 1)
    @test length(output3.coefs) == length(output1.coefs) - 1

    @test_throws AssertionError StateSpaceLearning.fit_model(y1; s = 200)
    @test_throws AssertionError StateSpaceLearning.fit_model(y1; model_type = "none")
    @test_throws AssertionError StateSpaceLearning.fit_model(y1; estimation_procedure = "none")
    @test_throws AssertionError StateSpaceLearning.fit_model(y1; α = -0.1)
    @test_throws AssertionError StateSpaceLearning.fit_model(y1; α = 1.1)

end

@testset "Function: forecast" begin
    y1 = rand(100)
    y2 = rand(100); y2[10:20] .= NaN

    output1 = StateSpaceLearning.fit_model(y1)
    @test length(StateSpaceLearning.forecast(output1, 10)) == 10

    output2 = StateSpaceLearning.fit_model(y2; Exogenous_X = rand(100, 3))
    @test length(StateSpaceLearning.forecast(output2, 10; Exogenous_Forecast = rand(10, 3))) == 10

    @test_throws AssertionError StateSpaceLearning.forecast(output1, -1)
    @test_throws AssertionError StateSpaceLearning.forecast(output1, 10; Exogenous_Forecast = rand(5, 3))
    @test_throws AssertionError StateSpaceLearning.forecast(output2, 10)
    @test_throws AssertionError StateSpaceLearning.forecast(output2, 10; Exogenous_Forecast = rand(5, 3))
end