function mse_func(prediction::Vector, test::Vector)
    return mean((prediction - test) .^ 2)
end

function bias_func(prediction::Vector, test::Vector)
    return mean((prediction - test))
end
