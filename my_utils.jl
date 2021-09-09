"""
    dncnn(; n_layers = 8)

build a DnCNN model as in:
> K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: 
> Residual Learning of Deep CNN for Image Denoising," in IEEE Transactions on 
> Image Processing, vol. 26, no. 7, pp. 3142-3155, July 2017.

"""
function dncnn(; n_layers = 8, bias = false)
    nn = Any[Conv((3, 3), 1 => 64, pad = 1, relu, bias=false)]
    for k in 1:n_layers - 2
        push!(nn, Conv((3, 3), 64 => 64, pad = 1, bias=false))
        push!(nn, BatchNorm(64, relu))
    end
    push!(nn, Chain(Conv((3, 3), 64 => 1, pad = 1, bias=bias)))
    Chain(nn...)
end

grid(Δ, n, T) = Δ .+ (0:floor(Int, (n - Δ)/T))*T

"""
    make_mask(img, T::Tuple{Int, Int}, Δ::Tuple{Int, Int})

make a grid mask for `img`. 
- `T` periods of the grid
- `Δ` initial shifts (≤ T).
"""
function make_mask(img, T::Tuple{Int, Int}, Δ::Tuple{Int, Int})
    
    any(Δ .> T) && error("check mask parameters")
    size_img = size(img)
    mask = fill(true, size_img)
    for k in grid(Δ[1], size_img[1], T[1]), l in grid(Δ[2], size_img[2], T[2])
        mask[k, l] = false
    end
    mask
end

"""
    make_mask(img, T::Tuple{Int, Int}, n::Int)

make a grid mask for `img`. 
- `T` periods of the grid
- `n` mask number.
"""
function make_mask(img, T::Tuple{Int, Int}, n::Int; target = gpu)
    
    (n > prod(T)) && error("check mask parameters")
    make_mask(img, T, (CartesianIndices(T)[n][1],CartesianIndices(T)[n][2]))
end


function make_data(x, T::Tuple{Int, Int})
    
    data = []
	for k in 1:T[1], l in 1:T[2]
        mask = make_mask(x, T, (k, l))
        data_in = x.*mask[:,:,:,:]
        data_ou = x.*(.!mask[:,:,:,:])    
        mask_ou = .!mask[:,:,:,:]    

        push!(data, (data_in, data_ou, mask_ou))
    end
    return data
end

"""
    J_func(train_data, m)

Compute the output of the J-invariant transformed model as defined in:
> Noise2Self: Blind Denoising by Self-Supervision. Joshua Batson, Loic Royer 
> Proceedings of the 36th International Conference on Machine Learning, PMLR 97:524-533, 2019.  
"""
function J_func(data, m)
    img = zeros(size(first(data)[1])) |> gpu
    for (data_in, data_ou, mask_ou) in data
        img += m(data_in).*mask_ou
    end
    img[:,:,1,1]
end

function sort_gpus()
	mems = Int[]
	gpus = collect(CUDA.devices())
	for k in 1:length(gpus)
		CUDA.device!(k-1)
		push!(mems, CUDA.available_memory())
	end
	ind = sortperm(mems, rev=true)
	hcat(gpus[ind], mems[ind])	
end

psnr(i, k) = 20*maximum(i) - 10*log10(sum((i - k).^2)/prod(size(k)))

imshow(x) = heatmap(x, yflip=true, aspect_ratio=1, cb = false, c=:greys, border=:none)
