module RingMatrices

"""
A Matrix with 1.0 along the -1 diagonal
"""
struct RingMatrix{T<:Real} <: AbstractMatrix{T}
    p::T
    n::Int64
    index::Vector{CartesianIndex}
    entries::Vector{T}
end

function RingMatrix(p::T, n) where T <: Real
    # compute the valid indices
    ni = (n-1) + 2
    indices = Vector{CartesianIndex}(undef, ni)
    entries = Vector{Float64}(undef, ni)
    k = 1
    for ii in CartesianIndices((1:n,1:n))
        if ii[1] == ii[2]==1
            indices[k] = ii
            entries[k] = 1-p
            k += 1
        elseif ii[1]==2&&ii[2]==1
            indices[k] = ii
            entries[k] = p 
            k += 1
        elseif ii[1]==ii[2]+1
            indices[k] = ii
            entries[k] = one(T)
            k+=1
        elseif ii[1]==1&&ii[2]==n
            indices[k] = ii
            entries[k] = one(T) 
        end
    end
    RingMatrix(p,n,indices,entries)
end

Base.size(X::RingMatrix{T}) where T <: Real = (X.n,X.n)

function Base.getindex(X::RingMatrix{T}, ii::CartesianIndex) where T <: Real    
    if ii in CartesianIndices((1:X.n, 1:X.n))
       jj = findfirst(k->k==ii, X.index)
       if jj === nothing
        return 0.0
       end
       X.entries[jj]
    end
end

function Base.getindex(X::RingMatrix{T}, i::Integer,j::Integer) where T <: Real
    if ((i < 1 || i > X.n) || (j < 1 || j > X.n))
        throw(BoundsError(X, [i,j]))
    end
    if i==j==1
        return 1-X.p
    end
    if i==1&&j==X.n
        return one(T)
    end
    if i==2&&j==1
        return X.p
    end
    if i==j+1
        return one(T)
    end
    return zero(T) 
end

"""
Product of two RingMatrices
"""
struct RingProductMatrix{T<:Real} <: AbstractMatrix{T}
    f1::RingMatrix{T}
    f2::RingMatrix{T}
    n::Int64
end

RingProductMatrix(f1::RingMatrix{T}, f2::RingMatrix{T}) where T <: Real = RingProductMatrix(f1,f2, f1.n*f2.n)
Base.size(X::RingProductMatrix{T}) where T <: Real = (X.n, X.n)

function Base.getindex(X::RingProductMatrix{T}, i::Integer, j::Integer) where T <: Real
    p1 = X.f1.p
    n1 = X.f1.n
    n2 = X.f2.n
    p2 = X.f2.p
    n = X.n
    n1n1 = n1*n1
    # convert from product index to individual index
    # find the block
    i1 = div(i-1,n2)+1
    j1 = div(j-1,n2)+1

    #find the index within the block
    i2 = i - (i1-1)*n2
    j2 = j - (j1-1)*n2

    X.f1[i1,j1]*X.f2[i2,j2]
end

struct PairwiseCombinations{T<:Real} <: AbstractMatrix{T}
    p::Vector{T}
    n::Int64
    K::Int64
    index::Vector{CartesianIndex}
    entries::Vector{T}
end

function PairwiseCombinations(X::Vector{RingMatrix{T}}) where T <: Real
    M = length(X)
    p = fill(zero(T), M)
    nn = fill(0,M)
    nnz = 1
    for (i,x) in enumerate(X)
        p[i] = x.p
        nn[i] = x.n
    end
    all(nn[1] .== nn) || error("All transition matrices should have the same number of states")
    
    K = length(X[1].entries)
    nnz = 1 # all silent state
    nnz += M*(K-1) # only one state active
    nnz += div(M*(M-1)*(K-1)^2,2) # all pairwise combinations

    offset = fill(0, M)
    for (i,n) in enumerate(nn)
        offset[i] = prod(nn[i+1:end]) 
    end
    entries = fill(zero(T), nnz)
    index = Vector{CartesianIndex}(undef, nnz)
    kk = 1
    entries[kk] = prod([x.entries[1] for x in X])
    index[kk] = CartesianIndex(1,1)
    kk += 1
    #all single activations 
    for i in 1:M
        x = X[i]
        _p = 1.0 
        for j in 1:M
            if j != i
                _p *= X[j].entries[1]
            end
        end
        offset1 = offset[i]
        for (k1,e1) in zip(x.index[2:end], x.entries[2:end])
            entries[kk] = _p*e1
            ii = (k1[1]-1)*offset1+1
            jj = (k1[2]-1)*offset1+1
            index[kk] = CartesianIndex(ii,jj)
            kk += 1
        end
    end

    #all pairwise combinations
    for i in 1:M-1
        for j in i+1:M 
            x1 = X[i]
            offset1 = offset[i]
            x2 = X[j]
            offset2 = offset[j]
            #TODO: Do not include the all-silent state here
            for (k1,e1) in zip(x1.index[2:end],x1.entries[2:end])
                for (k2,e2) in zip(x2.index[2:end], x1.entries[2:end])
                    ii = (k1[1]-1)*offset1+(k2[1]-1)*offset2+1
                    jj = (k1[2]-1)*offset1+(k2[2]-1)*offset2+1
                    index[kk] = CartesianIndex(ii,jj)
                    entries[kk] = e1*e2
                    kk += 1
                end
            end
        end
    end
    PairwiseCombinations(p, prod(nn), nn[1], index, entries)
end

Base.size(X::PairwiseCombinations{T}) where T <: Real = (X.n, X.n)

function Base.getindex(X::PairwiseCombinations{T}, ii::CartesianIndex) where T <: Real    
    if ii in CartesianIndices((1:X.n, 1:X.n))
       jj = findfirst(k->k==ii, X.index)
       if jj === nothing
        return 0.0
       end
       X.entries[jj]
    else
        throw(BoundsError(X, ii.I))
    end
end

Base.getindex(X::PairwiseCombinations{T},i::Integer, j::Integer) where T <: Real = getindex(X, CartesianIndex(i,j))
Base.getindex(X::PairwiseCombinations{T},i,j) where T <: Real = getindex(X,broadcast(CartesianIndex, i, permutedims(j)))

"""
```
function decompose(X::PairwiseCombinations{T}) where T <: Real
```
Decompose the combinations into their indiviual matrices
"""
function decompose(X::PairwiseCombinations{T}) where T <: Real
    [RingMatrix(_p, X.K) for _p in X.p]
end

end # module RingMatrices