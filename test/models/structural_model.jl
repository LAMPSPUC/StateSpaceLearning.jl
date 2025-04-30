@testset "StructuralModel" begin
    y1 = rand(100)

    model1 = StateSpaceLearning.StructuralModel(y1)
    model2 = StateSpaceLearning.StructuralModel(y1; freq_seasonal=[3, 10])
    model3 = StateSpaceLearning.StructuralModel(y1; cycle_period=[3, 10.2])
    model4 = StateSpaceLearning.StructuralModel(y1; cycle_period=[3, 10.2])

    @test typeof(model1) == StateSpaceLearning.StructuralModel
    @test typeof(model2) == StateSpaceLearning.StructuralModel
    @test typeof(model3) == StateSpaceLearning.StructuralModel
    @test typeof(model4) == StateSpaceLearning.StructuralModel

    @test_throws AssertionError StateSpaceLearning.StructuralModel(y1; stochastic_start=0)
    @test_throws AssertionError StateSpaceLearning.StructuralModel(
        y1; stochastic_start=1000
    )

    @test_throws AssertionError StateSpaceLearning.StructuralModel(y1; freq_seasonal=1000)

    exog_error = ones(100, 3)
    @test_throws AssertionError StateSpaceLearning.StructuralModel(y1; exog=exog_error)
end

@testset "create deterministic matrices" begin
    X1 = StateSpaceLearning.create_deterministic_cycle(100, 12)
    @test size(X1) == (100, 2)

    X2 = StateSpaceLearning.create_deterministic_seasonal(100, 12)
    @test size(X2) == (100, 12)

    X3 = StateSpaceLearning.create_initial_states_Matrix(
        100, [12, 20], true, true, true, true, [3, 10.2]
    )

    @test size(X3) == (100, 38)

    X4 = StateSpaceLearning.create_initial_states_Matrix(
        100, 12, true, true, true, false, 3
    )

    @test size(X4) == (100, 14)
end

@testset "Innovation matrices" begin
    @test StateSpaceLearning.ξ_size(10, 1) == 8
    @test StateSpaceLearning.ζ_size(10, 2, 1) == 6
    @test StateSpaceLearning.ζ_size(10, 0, 1) == 8
    @test StateSpaceLearning.ω_size(10, 2, 0, 1) == 9
    @test StateSpaceLearning.ω_size(10, 2, 2, 1) == 7
    @test StateSpaceLearning.o_size(10, 1) == 10
    @test StateSpaceLearning.ϕ_size(10, 0, 1) == 16
    @test StateSpaceLearning.ϕ_size(10, 2, 1) == 14

    @test StateSpaceLearning.ξ_size(10, 5) == 5
    @test StateSpaceLearning.ζ_size(10, 2, 5) == 3
    @test StateSpaceLearning.ζ_size(10, 0, 5) == 5
    @test StateSpaceLearning.ω_size(10, 2, 0, 5) == 6
    @test StateSpaceLearning.ω_size(10, 2, 2, 5) == 4
    @test StateSpaceLearning.o_size(10, 6) == 5
    @test StateSpaceLearning.ϕ_size(10, 0, 5) == 10

    X_ξ1 = StateSpaceLearning.create_ξ(5, 1)
    X_ξ2 = StateSpaceLearning.create_ξ(5, 3)

    @test X_ξ1 == [
        0.0 0.0 0.0
        1.0 0.0 0.0
        1.0 1.0 0.0
        1.0 1.0 1.0
        1.0 1.0 1.0
    ]

    @test X_ξ2 == [
        0.0 0.0
        0.0 0.0
        1.0 0.0
        1.0 1.0
        1.0 1.0
    ]

    X_ζ1 = StateSpaceLearning.create_ζ(5, 0, 1)
    X_ζ2 = StateSpaceLearning.create_ζ(6, 2, 1)
    X_ζ3 = StateSpaceLearning.create_ζ(5, 2, 3)

    @test X_ζ1 == [
        0.0 0.0 0.0
        0.0 0.0 0.0
        1.0 0.0 0.0
        2.0 1.0 0.0
        3.0 2.0 1.0
    ]

    @test X_ζ2 == [
        0.0 0.0
        0.0 0.0
        1.0 0.0
        2.0 1.0
        3.0 2.0
        4.0 3.0
    ]

    @test X_ζ3 == zeros(5, 0)

    X_ω1 = StateSpaceLearning.create_ω(5, 2, 0, 1)
    X_ω2 = StateSpaceLearning.create_ω(5, 2, 2, 1)
    X_ω3 = StateSpaceLearning.create_ω(5, 2, 0, 3)

    @test X_ω1 == [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        -1.0 1.0 0.0 0.0
        0.0 -1.0 1.0 0.0
        -1.0 1.0 -1.0 1.0
    ]

    @test X_ω2 == [
        0.0 0.0
        0.0 0.0
        -1.0 1.0
        0.0 -1.0
        -1.0 1.0
    ]

    @test X_ω3 == [
        0.0 0.0 0.0
        0.0 0.0 0.0
        1.0 0.0 0.0
        -1.0 1.0 0.0
        1.0 -1.0 1.0
    ]

    X_o1 = StateSpaceLearning.create_o_matrix(3, 1)
    X_o3 = StateSpaceLearning.create_o_matrix(3, 2)

    @test X_o1 == Matrix(1.0 * I, 3, 3)
    @test X_o3 == [
        0.0 0.0
        1.0 0.0
        0.0 1.0
    ]

    X_ϕ1 = StateSpaceLearning.create_ϕ(3, 5, 0, 1)
    X_ϕ2 = StateSpaceLearning.create_ϕ(3, 5, 3, 1)
    X_ϕ3 = StateSpaceLearning.create_ϕ(3, 5, 0, 2)

    @test X_ϕ1 == [
        0.0 0.0 0.0 0.0 0.0 0.0
        -0.5 -0.86603 0.0 0.0 0.0 0.0
        1.0 -0.0 1.0 -0.0 0.0 0.0
        -0.5 0.86603 -0.5 0.86603 -0.5 0.86603
        -0.5 -0.86603 -0.5 -0.86603 -0.5 -0.86603
    ]

    @test X_ϕ2 == [
        0.0 0.0
        -0.5 -0.86603
        1.0 -0.0
        -0.5 0.86603
        -0.5 -0.86603
    ]

    @test X_ϕ3 == [
        0.0 0.0 0.0 0.0 0.0 0.0
        -0.5 -0.86603 0.0 0.0 0.0 0.0
        1.0 -0.0 1.0 -0.0 0.0 0.0
        -0.5 0.86603 -0.5 0.86603 -0.5 0.86603
        -0.5 -0.86603 -0.5 -0.86603 -0.5 -0.86603
    ]
end

@testset "dynamic_exog_coefs" begin
    dynamic_exog_coefs1 = [(collect(1:5), "level")]

    X1 = StateSpaceLearning.create_dynamic_exog_coefs_matrix(
        dynamic_exog_coefs1, 5, 0, 0, 0, 1
    )
    @test size(X1) == (5, 4)

    dynamic_exog_coefs2 = [(collect(1:5), "slope")]

    X2 = StateSpaceLearning.create_dynamic_exog_coefs_matrix(
        dynamic_exog_coefs2, 5, 0, 0, 0, 1
    )
    @test size(X2) == (5, 4)

    dynamic_exog_coefs = [
        (collect(1:5), "level"),
        (collect(1:5), "slope"),
        (collect(1:5), "seasonal", 2),
        (collect(1:5), "cycle", 3),
    ]

    X = StateSpaceLearning.create_dynamic_exog_coefs_matrix(
        dynamic_exog_coefs, 5, 0, 0, 0, 1
    )
    @test size(X) == (5, 22)

    dynamic_exog_coefs = [
        (collect(6:7), "level", "", 4),
        (collect(6:7), "slope", "", 4),
        (collect(6:7), "seasonal", 2, 7),
        (collect(6:7), "cycle", 3, 8),
    ]
    X_f = StateSpaceLearning.create_forecast_dynamic_exog_coefs_matrix(
        dynamic_exog_coefs, 5, 2, 0, 0, 0, 1
    )
    @test size(X_f) == (2, 23)

    dynamic_exog_coefs2 = [(collect(6:7), "level", "", 4)]

    X_f2 = StateSpaceLearning.create_forecast_dynamic_exog_coefs_matrix(
        dynamic_exog_coefs2, 5, 2, 0, 0, 0, 1
    )
    @test size(X_f2) == (2, 4)

    dynamic_exog_coefs3 = [(collect(6:7), "slope", "", 4)]

    X_f3 = StateSpaceLearning.create_forecast_dynamic_exog_coefs_matrix(
        dynamic_exog_coefs3, 5, 2, 0, 0, 0, 1
    )
    @test size(X_f3) == (2, 4)
end

@testset "Create X matrix" begin
    exog1 = rand(5, 3)
    dynamic_exog_coefs = [
        (collect(1:5), "level"),
        (collect(1:5), "slope"),
        (collect(1:5), "seasonal", 2),
        (collect(1:5), "cycle", 3),
    ]

    X1 = StateSpaceLearning.create_X(
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        3,
        3,
        true,
        0,
        0,
        0,
        1,
        exog1,
        dynamic_exog_coefs,
    )

    exog2 = zeros(10, 3)
    dynamic_exog_coefs2 = [
        (collect(1:10), "level"),
        (collect(1:10), "slope"),
        (collect(1:10), "seasonal", 2),
        (collect(1:10), "cycle", 3),
    ]

    X2 = StateSpaceLearning.create_X(
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        3,
        3,
        true,
        3,
        2,
        4,
        1,
        exog2,
        dynamic_exog_coefs2,
    )

    @test size(X1) == (5, 52)
    @test size(X2) == (10, 85)
end

@testset "Function: get_components" begin
    exog1 = rand(10, 3)
    exog2 = zeros(10, 0)

    Basic_Structural = StateSpaceLearning.StructuralModel(
        rand(10); freq_seasonal=2, outlier=true, ζ_threshold=0, ω_threshold=0, exog=exog1
    )
    Local_Level = StateSpaceLearning.StructuralModel(
        rand(10);
        slope="none",
        seasonal="none",
        cycle="none",
        freq_seasonal=2,
        outlier=true,
        ζ_threshold=0,
        exog=exog1,
    )
    Local_Linear_Trend1 = StateSpaceLearning.StructuralModel(
        rand(10); seasonal="none", cycle="none", outlier=false, ζ_threshold=0, exog=exog1
    )
    Local_Linear_Trend2 = StateSpaceLearning.StructuralModel(
        rand(10);
        seasonal="none",
        cycle="none",
        freq_seasonal=2,
        outlier=false,
        ζ_threshold=0,
        exog=exog2,
    )

    models = [Basic_Structural, Local_Level, Local_Linear_Trend1, Local_Linear_Trend2]

    empty_keys_vec = [
        [], ["ν1", "ζ", "γ₁", "ω_2"], ["γ₁", "ω_2", "o"], ["γ₁", "ω_2", "o", "exog"]
    ]

    exogs = [exog1, exog1, exog1, exog2]

    for idx in eachindex(models)
        model = models[idx]

        components_indexes = StateSpaceLearning.get_components_indexes(model)

        for key in keys(components_indexes)
            @test (key in empty_keys_vec[idx]) ? isempty(components_indexes[key]) : true
            @test if key == "exog"
                length(components_indexes[key]) == size(exogs[idx], 2)
            else
                true
            end
        end
    end
end

@testset "Functions: get_variances and get_model_innovations" begin
    exog2 = zeros(10, 0)

    Basic_Structural = StateSpaceLearning.StructuralModel(
        rand(10); freq_seasonal=2, outlier=true, ζ_threshold=0, exog=exog2
    )
    Basic_Structural2 = StateSpaceLearning.StructuralModel(
        rand(10); freq_seasonal=[2, 5], outlier=true, ζ_threshold=0, exog=exog2
    )
    Local_Level = StateSpaceLearning.StructuralModel(
        rand(10);
        slope="none",
        seasonal="none",
        freq_seasonal=2,
        outlier=true,
        ζ_threshold=0,
        exog=exog2,
    )
    Local_Linear_Trend = StateSpaceLearning.StructuralModel(
        rand(10);
        seasonal="none",
        cycle="none",
        freq_seasonal=2,
        outlier=false,
        ζ_threshold=0,
        exog=exog2,
    )
    Local_Linear_Trend_cycle = StateSpaceLearning.StructuralModel(
        rand(10);
        seasonal="none",
        cycle="stochastic",
        freq_seasonal=2,
        cycle_period=3,
        outlier=false,
        ζ_threshold=0,
        exog=exog2,
    )

    models = [
        Basic_Structural,
        Basic_Structural2,
        Local_Level,
        Local_Linear_Trend,
        Local_Linear_Trend_cycle,
    ]

    params_vec = [
        ["ξ", "ζ", "ω_2", "ε"],
        ["ξ", "ζ", "ω_2", "ω_5", "ε"],
        ["ξ", "ε"],
        ["ξ", "ζ", "ε"],
        ["ξ", "ζ", "ε", "ϕ_3"],
    ]

    models_innovations = [
        ["ξ", "ζ", "ω_2"], ["ξ", "ζ", "ω_2", "ω_5"], ["ξ"], ["ξ", "ζ"], ["ξ", "ζ", "ϕ_3"]
    ]

    for idx in eachindex(models)
        model = models[idx]
        components_indexes = StateSpaceLearning.get_components_indexes(model)
        variances = StateSpaceLearning.get_variances(
            model, rand(100), rand(100), components_indexes
        )
        @test all([key in keys(variances) for key in params_vec[idx]])
        model_innovations = StateSpaceLearning.get_model_innovations(model)
        @test model_innovations == models_innovations[idx]
    end
end

@testset "Decomposion functions" begin
    Random.seed!(123)
    model = StateSpaceLearning.StructuralModel(
        vcat(collect(1:5), collect(5:-1:1));
        level="deterministic",
        seasonal="none",
        cycle="none",
        outlier=false,
        slope="stochastic",
        ζ_threshold=0,
    )
    StateSpaceLearning.fit!(model)
    slope = StateSpaceLearning.get_slope_decomposition(model, model.output.components)
    @test length(slope) == 10

    model = StateSpaceLearning.StructuralModel(
        vcat(rand(5) .+ 5, rand(5) .- 5) + vcat(collect(1:5), collect(5:-1:1));
        seasonal="none",
        cycle="none",
        outlier=false,
        slope="stochastic",
        ζ_threshold=0,
    )
    StateSpaceLearning.fit!(model)
    trend = StateSpaceLearning.get_trend_decomposition(
        model, model.output.components, slope
    )
    @test length(trend) == 10

    model = StateSpaceLearning.StructuralModel(
        rand(10);
        cycle="stochastic",
        cycle_period=3,
        outlier=false,
        slope="stochastic",
        ζ_threshold=0,
        freq_seasonal=3,
        ω_threshold=0,
        ϕ_threshold=0,
    )
    StateSpaceLearning.fit!(model)
    @test all(
        isapprox.(
            StateSpaceLearning.get_seasonal_decomposition(
                model, model.output.components, 3
            ),
            [
                -0.011114430313782316
                0.016993897901375513
                0.0
                -0.06137711460224293
                -0.028221145277986203
                0.045215043179361716
                -0.0027753371516487588
                -0.08682292272858037
                -0.036923166352424444
                0.13243351808078532
            ],
            atol=1e-6,
        ),
    )

    @test all(
        isapprox.(
            StateSpaceLearning.get_cycle_decomposition(model, model.output.components, 3),
            [
                0.0
                0.0
                1.6111635465327112e-18
                -0.005696779520104198
                -0.030765036256794644
                0.08224069806471179
                0.030974382058151183
                -0.15828117942894446
                0.037361403241860734
                0.17009485813575637
            ],
            atol=1e-6,
        ),
    )

    model_decomposition = StateSpaceLearning.get_model_decomposition(
        model, model.output.components
    )
    @test sort(collect(keys(model_decomposition))) ==
        sort(["cycle_3", "cycle_hat_3", "seasonal_3", "slope", "trend"])

    model = StateSpaceLearning.StructuralModel(
        vcat(rand(5) .+ 5, rand(5) .- 5) + vcat(collect(1:5), collect(5:-1:1));
        level="deterministic",
        seasonal="none",
        cycle="none",
        outlier=false,
        slope="stochastic",
        ζ_threshold=0,
    )
    StateSpaceLearning.fit!(model)
    trend = StateSpaceLearning.get_trend_decomposition(
        model, model.output.components, slope
    )
    @test length(trend) == 10

    model = StateSpaceLearning.StructuralModel(
        rand(100);
        cycle="stochastic",
        cycle_period=3,
        outlier=false,
        slope="stochastic",
        ζ_threshold=0,
        freq_seasonal=3,
        ω_threshold=0,
        ϕ_threshold=0,
        stochastic_start=30,
    )
    StateSpaceLearning.fit!(model)
    @test length(
        StateSpaceLearning.get_cycle_decomposition(model, model.output.components, 3)
    ) == 100
end

@testset "Function: simulate_states" begin
    model = StateSpaceLearning.StructuralModel(rand(100))
    StateSpaceLearning.fit!(model)
    @test length(StateSpaceLearning.simulate_states(model, 10, true, 12)) == 10
    @test length(StateSpaceLearning.simulate_states(model, 8, false, 12)) == 8
    @test length(StateSpaceLearning.simulate_states(model, 10, false, 0)) == 10

    model = StateSpaceLearning.StructuralModel(
        rand(100); seasonal="none", cycle="stochastic", cycle_period=3, outlier=false
    )
    StateSpaceLearning.fit!(model)
    @test length(StateSpaceLearning.simulate_states(model, 10, true, 12)) == 10
    @test length(StateSpaceLearning.simulate_states(model, 8, false, 12)) == 8
    @test length(StateSpaceLearning.simulate_states(model, 10, false, 0)) == 10
end

@testset "Function: forecast_dynamic_exog_coefs" begin
    Random.seed!(1234)
    model = StateSpaceLearning.StructuralModel(
        rand(100); seasonal="none", cycle="stochastic", cycle_period=3, outlier=false
    )
    StateSpaceLearning.fit!(model)
    @test StateSpaceLearning.forecast_dynamic_exog_coefs(
        model, 10, Vector{Vector}(undef, 0)
    ) == zeros(10)
    @test StateSpaceLearning.forecast_dynamic_exog_coefs(
        model, 8, Vector{Vector}(undef, 0)
    ) == zeros(8)

    dynamic_exog_coefs = [
        (collect(1:100), "level"),
        (collect(1:100), "slope"),
        (collect(1:100), "seasonal", 2),
        (collect(1:100), "cycle", 3),
    ]
    forecast_dynamic_exog_coefs = [
        collect(101:110), collect(101:110), collect(101:110), collect(101:110)
    ]
    model2 = StateSpaceLearning.StructuralModel(
        rand(100); dynamic_exog_coefs=dynamic_exog_coefs
    )
    StateSpaceLearning.fit!(model2)
    @test StateSpaceLearning.forecast_dynamic_exog_coefs(
        model2, 10, forecast_dynamic_exog_coefs
    ) != zeros(10)
    @test length(
        StateSpaceLearning.forecast_dynamic_exog_coefs(
            model2, 10, forecast_dynamic_exog_coefs
        ),
    ) == 10
end

@testset "Function: forecast" begin
    y1 = rand(100)
    y2 = rand(100)
    y2[10:20] .= NaN

    model1 = StateSpaceLearning.StructuralModel(y1)
    StateSpaceLearning.fit!(model1)
    @test length(StateSpaceLearning.forecast(model1, 10)) == 10

    model2 = StateSpaceLearning.StructuralModel(y2; exog=rand(100, 3))
    StateSpaceLearning.fit!(model2)
    @test length(StateSpaceLearning.forecast(model2, 10; Exogenous_Forecast=rand(10, 3))) ==
        10

    @test_throws AssertionError StateSpaceLearning.forecast(
        model1, 10; Exogenous_Forecast=rand(5, 3)
    )
    @test_throws AssertionError StateSpaceLearning.forecast(model2, 10)
    @test_throws AssertionError StateSpaceLearning.forecast(
        model2, 10; Exogenous_Forecast=rand(5, 3)
    )

    y3 = [
        4.718,
        4.77,
        4.882,
        4.859,
        4.795,
        4.905,
        4.997,
        4.997,
        4.912,
        4.779,
        4.644,
        4.77,
        4.744,
        4.836,
        4.948,
        4.905,
        4.828,
        5.003,
        5.135,
        5.135,
        5.062,
        4.89,
        4.736,
        4.941,
        4.976,
        5.01,
        5.181,
        5.093,
        5.147,
        5.181,
        5.293,
        5.293,
        5.214,
        5.087,
        4.983,
        5.111,
        5.141,
        5.192,
        5.262,
        5.198,
        5.209,
        5.384,
        5.438,
        5.488,
        5.342,
        5.252,
        5.147,
        5.267,
        5.278,
        5.278,
        5.463,
        5.459,
        5.433,
        5.493,
        5.575,
        5.605,
        5.468,
        5.351,
        5.192,
        5.303,
        5.318,
        5.236,
        5.459,
        5.424,
        5.455,
        5.575,
        5.71,
        5.68,
        5.556,
        5.433,
        5.313,
        5.433,
        5.488,
        5.451,
        5.587,
        5.594,
        5.598,
        5.752,
        5.897,
        5.849,
        5.743,
        5.613,
        5.468,
        5.627,
        5.648,
        5.624,
        5.758,
        5.746,
        5.762,
        5.924,
        6.023,
        6.003,
        5.872,
        5.723,
        5.602,
        5.723,
        5.752,
        5.707,
        5.874,
        5.852,
        5.872,
        6.045,
        6.142,
        6.146,
        6.001,
        5.849,
        5.72,
        5.817,
        5.828,
        5.762,
        5.891,
        5.852,
        5.894,
        6.075,
        6.196,
        6.224,
        6.001,
        5.883,
        5.736,
        5.82,
        5.886,
        5.834,
        6.006,
        5.981,
        6.04,
        6.156,
        6.306,
        6.326,
        6.137,
        6.008,
        5.891,
        6.003,
        6.033,
        5.968,
        6.037,
        6.133,
        6.156,
        6.282,
        6.432,
        6.406,
        6.23,
        6.133,
        5.966,
        6.068,
    ]
    model3 = StateSpaceLearning.StructuralModel(y3)
    StateSpaceLearning.fit!(model3)
    forecast3 = trunc.(StateSpaceLearning.forecast(model3, 18); digits=3)
    @assert forecast3 == [
        6.11,
        6.082,
        6.221,
        6.19,
        6.197,
        6.328,
        6.447,
        6.44,
        6.285,
        6.163,
        6.026,
        6.142,
        6.166,
        6.138,
        6.278,
        6.246,
        6.253,
        6.384,
    ]

    model4 = StateSpaceLearning.StructuralModel(y3; freq_seasonal=[12, 36])
    StateSpaceLearning.fit!(model4)
    forecast4 = trunc.(StateSpaceLearning.forecast(model4, 18); digits=3)

    @test length(forecast4) == 18

    exog = rand(length(y3), 3)
    model5 = StateSpaceLearning.StructuralModel(y3; exog=exog)
    StateSpaceLearning.fit!(model5)
    exog_forecast = rand(18, 3)
    forecast5 =
        trunc.(
            StateSpaceLearning.forecast(model5, 18; Exogenous_Forecast=exog_forecast);
            digits=3,
        )
    @test length(forecast5) == 18

    dynamic_exog_coefs = [
        (collect(1:length(y3)), "level"),
        (collect(1:length(y3)), "slope"),
        (collect(1:length(y3)), "seasonal", 2),
        (collect(1:length(y3)), "cycle", 3),
    ]
    forecast_dynamic_exog_coefs = [
        collect((length(y3) + 1):(length(y3) + 10)),
        collect((length(y3) + 1):(length(y3) + 10)),
        collect((length(y3) + 1):(length(y3) + 10)),
        collect((length(y3) + 1):(length(y3) + 10)),
    ]
    model6 = StateSpaceLearning.StructuralModel(y3; dynamic_exog_coefs=dynamic_exog_coefs)
    StateSpaceLearning.fit!(model6)
    @test StateSpaceLearning.forecast_dynamic_exog_coefs(
        model6, 10, forecast_dynamic_exog_coefs
    ) != zeros(10)
    @test length(
        StateSpaceLearning.forecast_dynamic_exog_coefs(
            model6, 10, forecast_dynamic_exog_coefs
        ),
    ) == 10

    model7 = StateSpaceLearning.StructuralModel(
        y3; seasonal="none", cycle="deterministic", cycle_period=[12, 6, 4, 3, 12 / 5, 2]
    )
    StateSpaceLearning.fit!(model7)
    forecast7 = trunc.(StateSpaceLearning.forecast(model7, 18); digits=3)
    @test length(forecast7) == 18

    model8 = StateSpaceLearning.StructuralModel(
        y3; seasonal="none", cycle="stochastic", cycle_period=[12, 6, 4, 3, 12 / 5, 2]
    )
    StateSpaceLearning.fit!(model8)
    forecast8 = trunc.(StateSpaceLearning.forecast(model8, 18); digits=3)
    @test length(forecast8) == 18
end

@testset "Function: simulate" begin
    y1 = rand(100)
    y2 = rand(100)
    y2[10:20] .= NaN

    model1 = StateSpaceLearning.StructuralModel(y1)
    StateSpaceLearning.fit!(model1)
    @test size(StateSpaceLearning.simulate(model1, 10, 100)) == (10, 100)

    @test size(
        StateSpaceLearning.simulate(model1, 10, 100; seasonal_innovation_simulation=10)
    ) == (10, 100)

    model2 = StateSpaceLearning.StructuralModel(y2; exog=rand(100, 3))
    StateSpaceLearning.fit!(model2)
    @test size(
        StateSpaceLearning.simulate(model2, 10, 100; Exogenous_Forecast=rand(10, 3))
    ) == (10, 100)

    exog = rand(length(y2), 3)
    model5 = StateSpaceLearning.StructuralModel(y2; exog=exog)
    StateSpaceLearning.fit!(model5)
    exog_forecast = rand(10, 3)
    @test size(
        StateSpaceLearning.simulate(model5, 10, 100; Exogenous_Forecast=exog_forecast)
    ) == (10, 100)

    dynamic_exog_coefs = [
        (collect(1:length(y2)), "level"),
        (collect(1:length(y2)), "slope"),
        (collect(1:length(y2)), "seasonal", 2),
        (collect(1:length(y2)), "cycle", 3),
    ]
    forecast_dynamic_exog_coefs = [
        collect((length(y2) + 1):(length(y2) + 10)),
        collect((length(y2) + 1):(length(y2) + 10)),
        collect((length(y2) + 1):(length(y2) + 10)),
        collect((length(y2) + 1):(length(y2) + 10)),
    ]
    model6 = StateSpaceLearning.StructuralModel(y2; dynamic_exog_coefs=dynamic_exog_coefs)
    StateSpaceLearning.fit!(model6)
    @test size(
        StateSpaceLearning.simulate(
            model6, 10, 100; dynamic_exog_coefs_forecasts=forecast_dynamic_exog_coefs
        ),
    ) == (10, 100)
end

@testset "Basics" begin
    model = StateSpaceLearning.StructuralModel(rand(100))
    @test StateSpaceLearning.isfitted(model) == false
    StateSpaceLearning.fit!(model)

    @test StateSpaceLearning.isfitted(model) == true
end
