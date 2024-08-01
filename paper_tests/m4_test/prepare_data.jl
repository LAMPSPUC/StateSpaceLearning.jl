function normalize(y::Vector, max_y::Float64, min_y::Float64)
    return (y .- min_y) ./ (max_y - min_y)
end

function de_normalize(y::Vector, max_y::Float64, min_y::Float64)
    return (y .* (max_y - min_y)) .+ min_y
end

function build_train_test_dict(df_train::DataFrame, df_test::DataFrame)
    train_test_dict = Dict()
    for i in eachindex(df_train[:, 1])
        key = df_train[:, 1][i]
        y_raw = Vector(df_train[i, :])[2:end]
        y_train_raw = y_raw[1:findlast(i->!ismissing(i), y_raw)]
        T = length(y_train_raw)
        y_train = y_train_raw
        y_test  = Vector(df_test[i, :])[2:end]

        y_max = maximum(y_train)
        y_min = minimum(y_train)

        y_train_normalized = normalize(y_train, y_max, y_min)

        train_test_dict[key] = Dict()
        train_test_dict[key]["normalized_train"] = y_train_normalized
        train_test_dict[key]["train"] = y_train
        train_test_dict[key]["test"]  = y_test
        train_test_dict[key]["max"]  = y_max
        train_test_dict[key]["min"]  = y_min
    end

    dict_vec = []
    for i in 1:48000
        key = "M$i"
        train_test_dict[key]["key"] = key
        push!(dict_vec, train_test_dict[key])
    end
    return dict_vec
end