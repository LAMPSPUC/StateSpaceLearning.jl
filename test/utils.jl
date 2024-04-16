@testset "Function: build_components" begin
    Exogenous_X1 = rand(10, 3)
    Exogenous_X2 = zeros(10, 0)

    parameter_combination = [
        ["Basic Structural", true, Exogenous_X1, []],
        ["Local Level", true, Exogenous_X1, ["ω", "γ₁", "ζ", "ν₁"]],
        ["Local Linear Trend", true, Exogenous_X1, ["ω", "γ₁"]],
        ["Basic Structural", false, Exogenous_X1, ["o"]],
        ["Basic Structural", true, Exogenous_X2, ["Exogenous_X"]],
    ]

    for param in parameter_combination
        X = StateSpaceLearning.create_X_unobserved_components(param[1], 10, 3, param[3], param[2], 0)
        components_indexes = StateSpaceLearning.get_components_indexes_unobserved_components(10, 3, param[3], param[2], param[1], 0)
        coefs = rand(size(X, 2))
        components = StateSpaceLearning.build_components(X, coefs, components_indexes)

        for key in keys(components)
            @test "Values" in keys(components[key])
            @test "Coefs" in keys(components[key])
            @test "Indexes" in keys(components[key])
            @test !(key in param[4]) ? !isempty(components[key]["Coefs"]) : isempty(components[key]["Coefs"])
            @test !(key in param[4]) ? !isempty(components[key]["Indexes"]) : isempty(components[key]["Indexes"])
            @test key == "Exogenous_X" ? "Selected" in keys(components[key]) : true
        end
    end
    
end

@testset "Function: get_fit_and_residuals" begin
    
    coefs = rand(10)
    X = rand(30, 10)
    T = 30

    valid_indexes1 = setdiff(collect(1:30), [11, 12])
    estimation_ϵ1 = rand(length(valid_indexes1))
    ϵ1, fitted1 = StateSpaceLearning.get_fit_and_residuals(estimation_ϵ1, coefs, X, valid_indexes1, T)
    @test all(isnan.(ϵ1[11:12]))
    @test !all(isnan.(ϵ1[valid_indexes1]))
    @test !all(isnan.(fitted1))

    valid_indexes2 = collect(1:30)
    estimation_ϵ2 = rand(length(valid_indexes2))
    ϵ2, fitted2 = StateSpaceLearning.get_fit_and_residuals(estimation_ϵ2, coefs, X, valid_indexes2, T)
    @test !all(isnan.(ϵ2[valid_indexes2]))
    @test !all(isnan.(fitted2))
end

@testset "Function: forecast_model" begin
    Exogenous_X = rand(30, 3)
    X = StateSpaceLearning.create_X_unobserved_components("Basic Structural", 30, 2, Exogenous_X, true, 0) 
    coefs = rand(size(X, 2))
    components_indexes  = StateSpaceLearning.get_components_indexes_unobserved_components(30, 2, Exogenous_X, true, "Basic Structural", 0)
    components          = StateSpaceLearning.build_components(X, coefs, components_indexes)

    output = StateSpaceLearning.Output("Basic Structural", X, coefs, rand(30), rand(30), components, Dict(), 3, 30, true, collect(1:30), 0, rand(60))
    @test length(StateSpaceLearning.unobserved_components_dict["forecast"](output, 5, rand(5, 3))) == 5
end