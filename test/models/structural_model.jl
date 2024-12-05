@testset "StructuralModel" begin
    y1 = rand(100)

    model1 = StateSpaceLearning.StructuralModel(y1)

    @test typeof(model1) == StateSpaceLearning.StructuralModel

    @test_throws AssertionError StateSpaceLearning.StructuralModel(y1; freq_seasonal=1000)

    exog_error = ones(100, 3)
    @test_throws AssertionError StateSpaceLearning.StructuralModel(
        y1; Exogenous_X=exog_error
    )
end

@testset "Innovation matrices" begin
    @test StateSpaceLearning.ξ_size(10) == 8
    @test StateSpaceLearning.ζ_size(10, 2) == 6
    @test StateSpaceLearning.ζ_size(10, 0) == 8
    @test StateSpaceLearning.ω_size(10, 2, 0) == 9
    @test StateSpaceLearning.ω_size(10, 2, 2) == 7
    @test StateSpaceLearning.o_size(10) == 10

    X_ξ1 = StateSpaceLearning.create_ξ(5, 0)
    X_ξ2 = StateSpaceLearning.create_ξ(5, 2)

    @test X_ξ1 == [
        0.0 0.0 0.0
        1.0 0.0 0.0
        1.0 1.0 0.0
        1.0 1.0 1.0
        1.0 1.0 1.0
    ]

    @test X_ξ2 == [
        0.0 0.0 0.0
        1.0 0.0 0.0
        1.0 1.0 0.0
        1.0 1.0 1.0
        1.0 1.0 1.0
        1.0 1.0 1.0
        1.0 1.0 1.0
    ]

    X_ζ1 = StateSpaceLearning.create_ζ(5, 0, 0)
    X_ζ2 = StateSpaceLearning.create_ζ(5, 2, 0)
    X_ζ3 = StateSpaceLearning.create_ζ(5, 2, 2)

    @test X_ζ1 == [
        0.0 0.0 0.0
        0.0 0.0 0.0
        1.0 0.0 0.0
        2.0 1.0 0.0
        3.0 2.0 1.0
    ]

    @test X_ζ2 == [
        0.0 0.0 0.0
        0.0 0.0 0.0
        1.0 0.0 0.0
        2.0 1.0 0.0
        3.0 2.0 1.0
        4.0 3.0 2.0
        5.0 4.0 3.0
    ]

    @test X_ζ3 == reshape(
        [
            0.0
            0.0
            1.0
            2.0
            3.0
            4.0
            5.0
        ],
        7,
        1,
    )

    X_ω1 = StateSpaceLearning.create_ω(5, 2, 0, 0)
    X_ω2 = StateSpaceLearning.create_ω(5, 2, 2, 0)

    @test X_ω1 == [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        -1.0 1.0 0.0 0.0
        0.0 -1.0 1.0 0.0
        -1.0 1.0 -1.0 1.0
    ]

    @test X_ω2 == [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        -1.0 1.0 0.0 0.0
        0.0 -1.0 1.0 0.0
        -1.0 1.0 -1.0 1.0
        0.0 -1.0 1.0 -1.0
        -1.0 1.0 -1.0 1.0
    ]

    X_o1 = StateSpaceLearning.create_o_matrix(3, 0)
    X_o2 = StateSpaceLearning.create_o_matrix(3, 2)

    @test X_o1 == Matrix(1.0 * I, 3, 3)
    @test X_o2 == vcat(Matrix(1.0 * I, 3, 3), zeros(2, 3))
end

@testset "Initial State Matrix" begin
    X1 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, true, true, true)
    X2 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, true, true, true)

    @test X1 == [
        1.0 0.0 1.0 0.0
        1.0 1.0 0.0 1.0
        1.0 2.0 1.0 0.0
        1.0 3.0 0.0 1.0
        1.0 4.0 1.0 0.0
    ]

    @test X2 == [
        1.0 0.0 1.0 0.0
        1.0 1.0 0.0 1.0
        1.0 2.0 1.0 0.0
        1.0 3.0 0.0 1.0
        1.0 4.0 1.0 0.0
        1.0 5.0 0.0 1.0
        1.0 6.0 1.0 0.0
    ]

    X3 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, true, true, false)
    X4 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, true, true, false)

    @test X3 == [
        1.0 0.0
        1.0 1.0
        1.0 2.0
        1.0 3.0
        1.0 4.0
    ]

    @test X4 == [
        1.0 0.0
        1.0 1.0
        1.0 2.0
        1.0 3.0
        1.0 4.0
        1.0 5.0
        1.0 6.0
    ]

    X5 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, true, false, false)
    X6 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, true, false, false)

    @test X5 == ones(5, 1)
    @test X6 == ones(7, 1)

    X7 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, false, true, false)
    X8 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, false, true, false)

    @test X7 == [0.0; 1.0; 2.0; 3.0; 4.0][:, :]
    @test X8 == [0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0][:, :]
end

@testset "Create X matrix" begin
    Exogenous_X1 = rand(5, 3)
    Exogenous_X2 = zeros(5, 0)

    Exogenous_forecast1 = rand(2, 3)

    param_combination = [
        Any[true, 0, 0],
        Any[true, 2, 0],
        Any[true, 2, 2],
        Any[false, 0, 0],
        Any[false, 2, 0],
        Any[false, 2, 2],
    ]

    size_vec1 = [
        (5, 22),
        (5, 18),
        (7, 18),
        (5, 17),
        (5, 13),
        (7, 13),
        (5, 12),
        (5, 12),
        (7, 12),
        (5, 7),
        (5, 7),
        (7, 7),
        (5, 16),
        (5, 14),
        (7, 14),
        (5, 11),
        (5, 9),
        (7, 9),
        (5, 13),
        (5, 11),
        (7, 11),
        (5, 8),
        (5, 6),
        (7, 6),
    ]
    size_vec2 = [
        (5, 19),
        (5, 15),
        (7, 15),
        (5, 14),
        (5, 10),
        (7, 10),
        (5, 9),
        (5, 9),
        (7, 9),
        (5, 4),
        (5, 4),
        (7, 4),
        (5, 13),
        (5, 11),
        (7, 11),
        (5, 8),
        (5, 6),
        (7, 6),
        (5, 10),
        (5, 8),
        (7, 8),
        (5, 5),
        (5, 3),
        (7, 3),
    ]
    counter = 1
    for args in [
        [true, true, true, true, true, true, 2, true, 12],
        [true, true, false, false, false, false, 2, true, 12],
        [true, true, true, true, false, false, 2, true, 12],
        [true, false, true, true, false, false, 2, true, 12],
    ]
        args = [x in [0, 1] ? Bool(x) : x for x in args]
        for param in param_combination
            args[8] = param[1]
            args[9] = param[2]
            if param[3] != 0
                X1 = StateSpaceLearning.create_X(
                    args..., Exogenous_X1, param[3], Exogenous_forecast1
                )
            else
                X1 = StateSpaceLearning.create_X(args..., Exogenous_X1, param[3])
            end
            X2 = StateSpaceLearning.create_X(args..., Exogenous_X2, param[3])
            @test size(X1) == size_vec1[counter]
            @test size(X2) == size_vec2[counter]
            counter += 1
        end
    end
end

@testset "Function: get_components" begin
    Exogenous_X1 = rand(10, 3)
    Exogenous_X2 = zeros(10, 0)

    Basic_Structural = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X1,
    )
    Local_Level = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=false,
        stochastic_trend=false,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X1,
    )
    Local_Linear_Trend1 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=false,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X1,
    )
    Local_Linear_Trend2 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=false,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X2,
    )

    models = [Basic_Structural, Local_Level, Local_Linear_Trend1, Local_Linear_Trend2]

    empty_keys_vec = [
        [], ["ν1", "ζ", "γ₁", "ω_2"], ["γ₁", "ω_2", "o"], ["γ₁", "ω_2", "o", "Exogenous_X"]
    ]

    exogs = [Exogenous_X1, Exogenous_X1, Exogenous_X1, Exogenous_X2]

    for idx in eachindex(models)
        model = models[idx]

        components_indexes = StateSpaceLearning.get_components_indexes(model)

        for key in keys(components_indexes)
            @test (key in empty_keys_vec[idx]) ? isempty(components_indexes[key]) : true
            @test if key == "Exogenous_X"
                length(components_indexes[key]) == size(exogs[idx], 2)
            else
                true
            end
        end
    end
end

@testset "Function: get_variances" begin
    Exogenous_X2 = zeros(10, 0)

    Basic_Structural = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X2,
    )
    Basic_Structural2 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=[2, 5],
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X2,
    )
    Local_Level = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=false,
        stochastic_trend=false,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X2,
    )
    Local_Linear_Trend = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=false,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X2,
    )

    models = [Basic_Structural, Basic_Structural2, Local_Level, Local_Linear_Trend]

    params_vec = [
        ["ξ", "ζ", "ω_2", "ε"], ["ξ", "ζ", "ω_2", "ω_5", "ε"], ["ξ", "ε"], ["ξ", "ζ", "ε"]
    ]

    for idx in eachindex(models)
        model = models[idx]
        components_indexes = StateSpaceLearning.get_components_indexes(model)
        variances = StateSpaceLearning.get_variances(
            model, rand(100), rand(100), components_indexes
        )
        @test all([key in keys(variances) for key in params_vec[idx]])
    end
end

@testset "Function: get_model_innovations" begin
    model1 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=zeros(10, 0),
    )
    model2 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=false,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=zeros(10, 0),
    )
    model3 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=false,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=zeros(10, 0),
    )
    model4 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=zeros(10, 0),
    )
    model5 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=false,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=[2, 5],
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=zeros(10, 0),
    )

    models = [model1, model2, model3, model4]

    keys_vec = [
        ["ξ", "ζ", "ω_2"], ["ζ", "ω_2"], ["ξ", "ω_2"], ["ξ", "ζ"], ["ξ", "ω_2", "ω_5"]
    ]

    for idx in eachindex(models)
        model = models[idx]
        model_innovations = StateSpaceLearning.get_model_innovations(model)
        @test model_innovations == keys_vec[idx]
    end
end

@testset "Function: get_innovation_simulation_X" begin
    innovation1 = "ξ"
    innovation2 = "ζ"
    innovation3 = "ω_2"

    model = StateSpaceLearning.StructuralModel(
        rand(3);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=zeros(10, 0),
    )
    steps_ahead = 2

    X1 = StateSpaceLearning.get_innovation_simulation_X(model, innovation1, steps_ahead)
    @assert X1 == [
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        1.0 1.0 0.0 0.0
        1.0 1.0 1.0 0.0
        1.0 1.0 1.0 1.0
        1.0 1.0 1.0 1.0
    ]

    X2 = StateSpaceLearning.get_innovation_simulation_X(model, innovation2, steps_ahead)
    @assert X2 == [
        0.0 0.0 0.0
        0.0 0.0 0.0
        1.0 0.0 0.0
        2.0 1.0 0.0
        3.0 2.0 1.0
        4.0 3.0 2.0
    ]

    X3 = StateSpaceLearning.get_innovation_simulation_X(model, innovation3, steps_ahead)
    @assert X3 == [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        -1.0 1.0 0.0 0.0
        0.0 -1.0 1.0 0.0
        -1.0 1.0 -1.0 1.0
        0.0 -1.0 1.0 -1.0
    ]
end
