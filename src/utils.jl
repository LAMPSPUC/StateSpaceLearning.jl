function build_components(X::Matrix{Tl}, coefs::Vector{Float64}, components_indexes::Dict{String, Vector{Int64}})::Dict where Tl
    components = Dict()
    for key in keys(components_indexes)
        components[key] = Dict()
        components[key]["Coefs"]   = coefs[components_indexes[key]]
        components[key]["Indexes"] = components_indexes[key]
        components[key]["Values"]  = X[:, components_indexes[key]]*coefs[components_indexes[key]]
    end
    components["Exogenous_X"]["Selected"] = findall(i -> i != 0, components["Exogenous_X"]["Coefs"])
    return components
end

function build_complete_variables(estimation_ϵ::Vector{Float64}, coefs::Vector{Float64}, X::Matrix{Tl}, valid_indexes::Vector{Int64}, T::Int64)::Tuple{Vector{Float64}, Vector{Float64}} where Tl
    ϵ      = fill(NaN, T); ϵ[valid_indexes] = estimation_ϵ
    fitted = X*coefs
    return ϵ, fitted
end