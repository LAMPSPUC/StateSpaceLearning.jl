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
        X = StateSpaceLearning.create_X(param[1], 10, 3, param[3], param[2], 0)
        components_indexes = StateSpaceLearning.get_components_indexes(10, 3, param[3], param[2], param[1], 0)
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

@testset "Function: build_complete_variables" begin
    
    coefs = rand(10)
    X = rand(30, 10)
    T = 30

    valid_indexes1 = setdiff(collect(1:30), [11, 12])
    estimation_ϵ1 = rand(length(valid_indexes1))
    ϵ1, fitted1 = StateSpaceLearning.build_complete_variables(estimation_ϵ1, coefs, X, valid_indexes1, T)
    @test all(isnan.(ϵ1[11:12]))
    @test !all(isnan.(ϵ1[valid_indexes1]))
    @test !all(isnan.(fitted1))

    valid_indexes2 = collect(1:30)
    estimation_ϵ2 = rand(length(valid_indexes2))
    ϵ2, fitted2 = StateSpaceLearning.build_complete_variables(estimation_ϵ2, coefs, X, valid_indexes2, T)
    @test !all(isnan.(ϵ2[valid_indexes2]))
    @test !all(isnan.(fitted2))
end