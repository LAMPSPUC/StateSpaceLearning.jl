module PlotsExt

using StateSpaceLearning: StateSpaceLearning

using Plots

function StateSpaceLearning.plot_point_forecast(y::Vector, prediction::Vector)
    T = length(y)
    steps_ahead = length(prediction)
    p = plot(collect(1:T), y; w=2, color="Black", lab="Historical", legend=:outerbottom)
    plot!(collect((T + 1):(T + steps_ahead)), prediction; lab="Forecast", w=2, color="blue")
    return p
end

function StateSpaceLearning.plot_scenarios(y::Vector, simulation::Matrix)
    T = length(y)
    steps_ahead, n_scenarios = size(simulation)
    p = plot(collect(1:T), y; w=2, color="Black", lab="Historical", legend=:outerbottom)
    for s in 1:(n_scenarios - 1)
        plot!(
            collect((T + 1):(T + steps_ahead)), simulation[:, s]; lab="", α=0.1, color="red"
        )
    end
    plot!(
        collect((T + 1):(T + steps_ahead)),
        simulation[:, n_scenarios];
        lab="Scenarios Paths",
        α=0.1,
        color="red",
    )
    return p
end

end
