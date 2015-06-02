require "torch"
require "nn"

dofile "avg_word_model.lua"


saved_dir = "./saved_model/"
dev_file = "../data/dev/validation_set_processed.tab"
dev_out = "../data/dev/dev_out/"
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a "'..directory..'"'):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end


files = scandir("./saved_model/")
scores = {}
for i, f in ipairs(files) do
	if string.find(f, ".net") then
		mod = torch.load(saved_dir ..f)
		mod:autotest(dev_file, dev_out ..f .. ".out")
	end
end
