function bias_func(prediction::Vector, test::Vector)
    return mean((prediction - test))
end
