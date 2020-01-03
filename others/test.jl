using NIfTI
include("utils.jl")

PATH = "/media/miguelv/HD1/Inconsistency/baseline/"
PATH_DATA = "/media/miguelv/HD1/CR_DATA/"

path = string(PATH, "1/preds/02NOV2016_24h_12_logits.nii.gz")

println(path)

logits = niread(path);
softed = softmax(logits);
arged = my_argmax(softed);
#niwrite("lala.nii.gz", NIVolume(arged))

"""
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
			GT = cat(GT .== 0, GT .== 1, dims=length(size(GT))+1)

			# Reading values
			softed = softmax(niread(filepath));
			y_pred = my_argmax(softed);
			println(sum(y_pred .== 1))
			

			# Calculate DICE and other metrics
			dice(y_pred, GT)

		end
	end
end
"""
