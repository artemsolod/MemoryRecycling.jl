module MemoryRecycling
import Mmap: MADV_HUGEPAGE

const mem_cache = (; lock=Threads.SpinLock(), dict=Dict{Tuple{DataType, Int}, Vector{Any}}())
const cache_on = Base.ScopedValues.ScopedValue(false) # toggle array caching

function dropcache!()
    Base.ScopedValues.with(cache_on => false) do
        empty!(mem_cache.dict)
    end
end

function cached_mem_allocator(::Type{T}, nels) where T
    mem = lock(mem_cache.lock) do
        cache_vec = get(mem_cache.dict, (T, nels), nothing)
        if !isnothing(cache_vec) && !isempty(cache_vec)
            pop!(cache_vec) :: Memory{T}
        else
            nbytes = sizeof(T)*nels
            if Sys.islinux() & (nbytes >= 2^20)
                ptr = @ccall pvalloc(nbytes::Csize_t)::Ptr{Cvoid} # depreceated pvalloc
                retcode = @ccall madvise(ptr::Ptr{Cvoid}, nbytes::Csize_t, MADV_HUGEPAGE::Cint)::Cint
                @assert iszero(retcode)           
            else
                ptr = Libc.malloc(nbytes)
            end
            @ccall jl_ptr_to_genericmemory(Memory{T}::Any, ptr::Ptr{Cvoid}, nels::Csize_t, 1::Cint)::Memory{T}
        end
    end
    return finalizer(_finalize_mem, mem)
end

function _finalize_mem(m :: Memory{T}) where T
    if islocked(mem_cache.lock) || !trylock(mem_cache.lock)
        finalizer(_finalize_mem, m)
        return nothing
    end
    try
        cache_vec = get!(() -> Vector{Memory{T}}(), mem_cache.dict, (T, length(m)))
        push!(cache_vec, m)
    finally
        unlock(mem_cache.lock)
    end
end

for T in (Float64, Float32, Float16, Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt8)
    @eval begin
        @inline function Memory{$T}(::UndefInitializer, m::Int64)
            if cache_on[] && (m >= 100_000)
                cached_mem_allocator($T, m)
            else
                @ccall jl_alloc_genericmemory(Memory{$T}::Any, m::Csize_t)::Memory{$T}
            end
        end
    end
end
end # module
