function softmax(logits)
	e_x = exp.(logits .+ 1e-20)
	suma = sum(e_x, dims=length(size(e_x)))
	return e_x ./ (suma .+ 1e-20)
end

function my_argmax(softed)
	# Axis is always -1
	# softed: 256, 256, 18, 2
	# returns: 256, 256, 18
	println(size(softed))
	idx = argmax(softed, dims=4)
	return cat(zeros(256, 256, 18), ones(256, 256, 18), dims=4)[idx][:,:,:,1]
end

function dice(y_pred, y_true)
	# y_pred: 256, 256, 18
	# y_pred: 256, 256, 18
	if sum(y_true) == 0
		if sum(y_pred) == 1
			result = 1
		else
			result = 1.0 * (sum(y_true .== 0) - sum(y_pred)) / sum( y_true .== 0)
		end
	else
		num = 2 * sum(y_pred .* y_true)
		denom = sum(y_pred) + sum(y_true)
		result = num / denom
	end

	return result
end
