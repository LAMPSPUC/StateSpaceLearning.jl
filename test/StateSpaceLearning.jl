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

    output3 = StateSpaceLearning.fit_model(y1; ζ_ω_threshold = 1)
    @test length(output3.coefs) - 22 == length(output1.coefs)

    @test_throws AssertionError StateSpaceLearning.fit_model(y1; model_input = Dict("stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 1000))
   
    @test_throws AssertionError StateSpaceLearning.fit_model(y1; estimation_input = Dict("α" => -0.1, "information_criteria" => "aic", "ψ" => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true))
    @test_throws AssertionError StateSpaceLearning.fit_model(y1; estimation_input = Dict("α" => 1.1, "information_criteria" => "aic", "ψ" => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true))

end

@testset "Function: forecast" begin
    y1 = rand(100)
    y2 = rand(100); y2[10:20] .= NaN

    output1 = StateSpaceLearning.fit_model(y1)
    @test length(StateSpaceLearning.forecast(output1, 10)) == 10

    output2 = StateSpaceLearning.fit_model(y2; Exogenous_X = rand(100, 3))
    @test length(StateSpaceLearning.forecast(output2, 10; Exogenous_Forecast = rand(10, 3))) == 10

    @test_throws AssertionError StateSpaceLearning.forecast(output1, 10; Exogenous_Forecast = rand(5, 3))
    @test_throws AssertionError StateSpaceLearning.forecast(output2, 10)
    @test_throws AssertionError StateSpaceLearning.forecast(output2, 10; Exogenous_Forecast = rand(5, 3))
end