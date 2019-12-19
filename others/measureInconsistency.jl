function softmax(logits)
	e_x = exp(logits+1e-10)
	return e_x / expand_dims.. sum(e_x, dims=1)+1e-10
end

PATH = "/media/miguelv/HD1/Inconsistency/baseline/"
PATH_DATA = "/media/miguelv/HD1/CR_DATA/"

# PART 1. PERFORMANCE WRT. GROUND TRUTH

println(softmax(10))
