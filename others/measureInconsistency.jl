using NIfTI
using ImageDistances

include("utils.jl")

PATH = "/media/miguelv/HD1/Inconsistency/baseline/"
PATH_DATA = "/media/miguelv/HD1/CR_DATA/"

# PART 1. PERFORMANCE WRT. GROUND TRUTH
println("Performance assessment")
for i=1:length(readdir(PATH))
	# I go inside every runa
	println(string("Run ", i))
	if isdir(string(PATH, i))
		for f in readdir(string(PATH, i, "/preds/"))
			filepath = string(PATH, i, "/preds/", f)
			println(filepath)

			# Reading and setting up the Ground Truth
			splitted_f = split(f, "_");
			GTpath = string(PATH_DATA, splitted_f[1], "/", splitted_f[2], "/", splitted_f[3], "/")
			if isfile(string(GTpath, "scan_miguel.nii.gz"));
				GT = niread(string(GTpath, "scan_miguel.nii.gz"));
			elseif isfile(string(GTpath, "scan_lesion.nii.gz"))
				GT = niread(string(GTpath, "scan_lesion.nii.gz"));
			else
				GT = zeros(256, 256, 18);
			end
			#GT = cat(GT .== 0, GT .== 1, dims=length(size(GT))+1)

			# Reading values
			softed = softmax(niread(filepath));
			y_pred = my_argmax(softed);

			println(size(y_pred))
			

			# Calculate DICE and other metrics
			println(dice(y_pred, GT))
			if sum(GT) != 0
				@time begin
					println(hausdorff(y_pred, GT))
				end
			end

		end
	end
end

