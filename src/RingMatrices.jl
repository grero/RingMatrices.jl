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

end # module RingMatrices
